use std::collections::HashMap;
use std::future::Future;
use std::hash::Hash;
use std::sync::Arc;

use tokio::sync::{broadcast, Mutex};

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
    in_flight: Mutex<HashMap<K, broadcast::Sender<V>>>,
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
    /// the task itself died. The work having already finished is not that case: it is published
    /// before the key is released, so a late subscriber either receives it or starts afresh.
    pub async fn run<F, Fut>(self: &Arc<Self>, key: K, work: F) -> Option<V>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = V> + Send,
    {
        let mut receiver = {
            let mut in_flight = self.in_flight.lock().await;
            match in_flight.get(&key) {
                Some(sender) => sender.subscribe(),
                None => {
                    // Capacity one: every receiver is waiting for the same single value.
                    let (sender, receiver) = broadcast::channel(1);
                    in_flight.insert(key.clone(), sender.clone());
                    let this = Arc::clone(self);
                    let task_key = key.clone();
                    tokio::spawn(async move {
                        let value = work().await;
                        // Published before the key is released. The other order would let a caller
                        // arriving in between subscribe to a channel that will never speak.
                        let _ = sender.send(value);
                        this.in_flight.lock().await.remove(&task_key);
                    });
                    receiver
                }
            }
        };

        receiver.recv().await.ok()
    }

    /// How many pieces of work are running. For tests and for reporting.
    pub async fn len(&self) -> usize {
        self.in_flight.lock().await.len()
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
            if flight.len().await == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(flight.len().await, 0, "the key should not be held");

        // And a second run genuinely re-runs rather than hanging.
        assert_eq!(flight.run(3, || async { 2 }).await, Some(2));
    }
}
