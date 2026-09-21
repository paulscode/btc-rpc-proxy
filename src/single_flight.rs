use std::collections::HashMap;
use std::future::Future;
use std::hash::Hash;
use std::sync::{Arc, Mutex, PoisonError};

use tokio::sync::broadcast;

/// Runs one piece of work per key, detached from whoever asked for it.
///
/// Two properties, both of which the peer-fetch path needs and neither of which a plain `await`
/// gives it:
///
/// * **The work outlives its caller.** It runs as its own task, so a caller that goes away
///   mid-flight does not take the work with it. For a block fetch that is the difference between a
///   request which fails forever and one which fails once and is served from cache on the retry:
///   the fetch completes and fills the cache regardless of who is still listening.
/// * **Concurrent callers share it.** The second caller for a key already in flight waits on the
///   same result rather than starting a second fetch, which matters as soon as one client can ask
///   for several blocks at once.
///
/// Values are published over a broadcast channel rather than returned, because there may be no
/// receiver left by the time there is something to return, and that is the expected case rather
/// than an error.
impl<K, V> std::fmt::Debug for SingleFlight<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The map is behind an async lock, so its contents are not reachable from a sync
        // formatter. The type is all `State`'s derived `Debug` needs from it.
        f.write_str("SingleFlight")
    }
}

pub struct SingleFlight<K, V> {
    // A plain mutex, not an async one: nothing in any critical section below awaits, and a
    // blocking guard is what lets the key be released from `Drop` when a task unwinds.
    in_flight: Mutex<HashMap<K, broadcast::Sender<V>>>,
}

/// Releases a key however the task holding it ends, including by panicking.
///
/// Without this a panic anywhere in `work` would leave the key in the map, and the map holds a
/// *clone* of the sender. The task's own copy drops as it unwinds, but the map's does not, so the
/// channel never closes: every later caller for that key finds the entry, subscribes, and then
/// waits on it **forever** rather than being told anything. Measured by neutering this guard --
/// the test below stops failing and starts hanging.
///
/// One panic would therefore cost one block, permanently and silently, for the life of the
/// process. Too wide a gap between cause and effect to leave open, with the work driven by bytes
/// from peers.
struct ReleaseOnDrop<K: Eq + Hash, V> {
    flight: Arc<SingleFlight<K, V>>,
    key: Option<K>,
}

impl<K: Eq + Hash, V> Drop for ReleaseOnDrop<K, V> {
    fn drop(&mut self) {
        // `Some` only when the task did not publish, so this never removes an entry a later run
        // put there: the publishing path takes the key out of the guard under the same lock.
        if let Some(key) = self.key.take() {
            self.flight
                .in_flight
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&key);
        }
    }
}

impl<K, V> Default for SingleFlight<K, V> {
    fn default() -> Self {
        SingleFlight {
            in_flight: Mutex::new(HashMap::new()),
        }
    }
}

impl<K, V> SingleFlight<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + 'static,
{
    /// Waits for `work` on `key`, starting it only if it is not already running.
    ///
    /// Returns `None` if the work produced no value for this caller to see, which happens only if
    /// the task itself died. The work having already finished is not that case: publishing the
    /// value and releasing the key happen under one lock, so a caller either subscribes in time to
    /// receive it or finds no key and starts afresh, never in between.
    pub async fn run<F, Fut>(self: &Arc<Self>, key: K, work: F) -> Option<V>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = V> + Send,
    {
        let mut receiver = {
            let mut in_flight = self.in_flight.lock().unwrap_or_else(PoisonError::into_inner);
            match in_flight.get(&key) {
                Some(sender) => sender.subscribe(),
                None => {
                    // Capacity one: every receiver is waiting for the same single value.
                    let (sender, receiver) = broadcast::channel(1);
                    in_flight.insert(key.clone(), sender.clone());
                    let this = Arc::clone(self);
                    let task_key = Some(key.clone());
                    tokio::spawn(async move {
                        let mut release = ReleaseOnDrop {
                            flight: Arc::clone(&this),
                            key: task_key,
                        };
                        let value = work().await;
                        // Publish and release the key under one lock, so a caller either finds the
                        // key before the value is published and receives it, or finds no key and
                        // starts afresh. Doing these as two steps leaves a window in between where
                        // a caller subscribes to a channel that has already spoken and then waits
                        // on it forever: a broadcast receiver only sees values sent after it
                        // subscribed. That is not theoretical — on a multi-threaded runtime it hit
                        // ~2.5% of callers under contention, each of them told the block could not
                        // be fetched when it had in fact just been fetched successfully.
                        let mut in_flight =
                            this.in_flight.lock().unwrap_or_else(PoisonError::into_inner);
                        let _ = sender.send(value);
                        if let Some(key) = release.key.take() {
                            in_flight.remove(&key);
                        }
                    });
                    receiver
                }
            }
        };

        receiver.recv().await.ok()
    }

    /// How many pieces of work are running. For tests and for reporting.
    pub fn len(&self) -> usize {
        self.in_flight
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::SingleFlight;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// The property the block cache depends on: a caller that gives up does not cancel the work.
    /// Before this, a fetch abandoned at the timeout was thrown away, so the next request started
    /// from nothing and failed the same way forever.
    #[tokio::test]
    async fn work_outlives_a_caller_that_goes_away() {
        let flight: Arc<SingleFlight<u8, u8>> = Arc::default();
        let ran = Arc::new(AtomicUsize::new(0));

        let counter = ran.clone();
        let abandoned = flight.run(1, move || async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            counter.fetch_add(1, Ordering::SeqCst);
            42
        });
        // Give up well before the work could finish, exactly as a timeout would.
        assert!(
            tokio::time::timeout(Duration::from_millis(5), abandoned)
                .await
                .is_err()
        );

        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(ran.load(Ordering::SeqCst), 1, "the work should have run on");
    }

    /// Concurrent callers for the same key share one run rather than each starting their own.
    #[tokio::test]
    async fn concurrent_callers_share_one_run() {
        let flight: Arc<SingleFlight<u8, u8>> = Arc::default();
        let ran = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];
        for _ in 0..5 {
            let flight = flight.clone();
            let counter = ran.clone();
            handles.push(tokio::spawn(async move {
                flight
                    .run(7, move || async move {
                        tokio::time::sleep(Duration::from_millis(30)).await;
                        counter.fetch_add(1, Ordering::SeqCst);
                        9
                    })
                    .await
            }));
        }

        for h in handles {
            assert_eq!(h.await.unwrap(), Some(9), "every caller gets the value");
        }
        assert_eq!(ran.load(Ordering::SeqCst), 1, "but it only ran once");
    }

    /// A caller arriving just as the work finishes must not be told the work failed.
    ///
    /// Publishing the value and releasing the key used to be two steps, which left a window where
    /// a caller found the key, subscribed, and then waited on a broadcast channel that had already
    /// spoken -- a receiver only sees values sent after it subscribed. Those callers got `None`,
    /// which `fetch_block_raw` reports as "block not available" for a block that had in fact just
    /// been fetched. Hammering the window this way produced it in ~2.5% of calls before the fix.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_caller_arriving_as_the_work_finishes_still_gets_the_value() {
        let missed = Arc::new(AtomicUsize::new(0));

        for round in 0..300u32 {
            let flight: Arc<SingleFlight<u32, u32>> = Arc::default();
            let mut handles = vec![];
            for _ in 0..8 {
                let flight = flight.clone();
                let missed = missed.clone();
                handles.push(tokio::spawn(async move {
                    for _ in 0..20 {
                        // The work always succeeds, so every `None` here is the race and not a
                        // failure being reported honestly.
                        let got = flight
                            .run(round, move || async move {
                                tokio::task::yield_now().await;
                                round
                            })
                            .await;
                        if got.is_none() {
                            missed.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        }

        let n = missed.load(Ordering::SeqCst);
        assert_eq!(n, 0, "{} callers were told the work failed when it had not", n);
    }

    /// A run whose work panics must still release its key.
    ///
    /// The caller seeing `None` is honest -- no value was produced. What must not happen is the
    /// key being left in the map. The map holds a clone of the sender, so the channel would never
    /// close, and the second call below would not fail but hang: the failure mode is a wedged
    /// caller, not an error. Neuter `ReleaseOnDrop` and this test stops finishing at all.
    ///
    /// The panic is inside a spawned task, so tokio logs it and the test itself survives; the
    /// noise in the output is expected.
    #[tokio::test]
    async fn a_run_whose_work_panics_releases_its_key() {
        let flight: Arc<SingleFlight<u8, u8>> = Arc::default();

        assert_eq!(
            flight.run(5, || async { panic!("work blew up") }).await,
            None,
            "nothing was produced, so there is nothing to hand back"
        );

        // The release happens as the task unwinds, so allow it to be scheduled.
        for _ in 0..100 {
            if flight.len() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(flight.len(), 0, "a panicking run must not keep its key");

        assert_eq!(
            flight.run(5, || async { 7 }).await,
            Some(7),
            "and the next caller for it gets a real answer rather than a dead channel"
        );
    }

    /// Different keys do not share, or one slow block would hold up every other.
    #[tokio::test]
    async fn different_keys_run_separately() {
        let flight: Arc<SingleFlight<u8, u8>> = Arc::default();
        let ran = Arc::new(AtomicUsize::new(0));

        for key in 0..3u8 {
            let counter = ran.clone();
            assert_eq!(
                flight
                    .run(key, move || async move {
                        counter.fetch_add(1, Ordering::SeqCst);
                        key
                    })
                    .await,
                Some(key)
            );
        }
        assert_eq!(ran.load(Ordering::SeqCst), 3);
    }

    /// The key is released once the work is done, so a later caller starts a fresh run rather than
    /// waiting on a channel nobody will ever speak on again.
    #[tokio::test]
    async fn the_key_is_released_when_the_work_finishes() {
        let flight: Arc<SingleFlight<u8, u8>> = Arc::default();
        assert_eq!(flight.run(3, || async { 1 }).await, Some(1));

        // The removal happens in the task just after publishing, so allow it to be scheduled.
        for _ in 0..100 {
            if flight.len() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(flight.len(), 0, "the key should not be held");

        // And a second run genuinely re-runs rather than hanging.
        assert_eq!(flight.run(3, || async { 2 }).await, Some(2));
    }
}

