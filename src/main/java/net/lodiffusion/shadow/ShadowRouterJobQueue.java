package net.lodiffusion.shadow;

import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Thread-safe job queue for LOD terrain generation requests from Voxy.
 * 
 * Maintains separate per-LOD priority queues (LOD 0–4) ordered by distance to player.
 * Supports enqueue (from mixin callback) and dequeue (from dispatcher).
 */
@SuppressWarnings("unchecked")
public class ShadowRouterJobQueue {

    private record RequestKey(int lod, int x, int y, int z) {
        static RequestKey of(VoxyRequestDecoder.VoxyNodeRequest req) {
            return new RequestKey(req.lodLevel, req.worldX, req.worldY, req.worldZ);
        }
    }
    
    private static final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    @SuppressWarnings("unchecked")
    private static final PriorityQueue<VoxyRequestDecoder.VoxyNodeRequest>[] lodQueues = 
        new PriorityQueue[5];

    // Player position in section-space units (updated by LodGenerationService).
    private static volatile int playerSectionX = 0;
    private static volatile int playerSectionZ = 0;

    // Requests currently queued but not yet handed out.
    private static final Set<RequestKey> queuedKeys = new HashSet<>();
    // Requests handed out by dequeue* and awaiting completion callback.
    private static final Set<RequestKey> inFlightKeys = new HashSet<>();
    
    static {
        for (int i = 0; i < 5; i++) {
            // Order by ascending distance: closest requests first
            ShadowRouterJobQueue.lodQueues[i] = new PriorityQueue<>(
                Comparator.comparingDouble(ShadowRouterJobQueue::estimateDistance)
            );
        }
    }
    
    /**
     * Enqueue a single request (called by VoxyShadowBridgeMixin).
     * 
     * @param request Decoded Voxy node request
     */
    public static void enqueue(VoxyRequestDecoder.VoxyNodeRequest request) {
        if (request == null || request.lodLevel < 0 || request.lodLevel > 4) {
            return;  // Ignore invalid requests
        }
        
        lock.writeLock().lock();
        try {
            RequestKey key = RequestKey.of(request);
            if (queuedKeys.contains(key) || inFlightKeys.contains(key)) {
                return;
            }
            lodQueues[request.lodLevel].offer(request);
            queuedKeys.add(key);
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Enqueue multiple requests (batch optimization).
     * 
     * @param requests Array of requests to enqueue
     */
    public static void enqueueBatch(VoxyRequestDecoder.VoxyNodeRequest[] requests) {
        if (requests == null || requests.length == 0) {
            return;
        }
        
        lock.writeLock().lock();
        try {
            for (VoxyRequestDecoder.VoxyNodeRequest req : requests) {
                if (req != null && req.lodLevel >= 0 && req.lodLevel <= 4) {
                    RequestKey key = RequestKey.of(req);
                    if (queuedKeys.contains(key) || inFlightKeys.contains(key)) {
                        continue;
                    }
                    lodQueues[req.lodLevel].offer(req);
                    queuedKeys.add(key);
                }
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Dequeue the highest-priority request across all LOD levels.
     * Priority: closest distance, prefer higher LOD (coarser first) for efficiency.
     * 
     * @return Next request to generate, or null if queue is empty
     */
    public static VoxyRequestDecoder.VoxyNodeRequest dequeueAny() {
        lock.writeLock().lock();
        try {
            // Sort by distance; prefer finer LOD (smaller level number) on ties.
            VoxyRequestDecoder.VoxyNodeRequest best = null;
            int bestLod = Integer.MAX_VALUE;
            double bestDist = Double.MAX_VALUE;
            
            for (int lod = 0; lod <= 4; lod++) {
                if (!lodQueues[lod].isEmpty()) {
                    VoxyRequestDecoder.VoxyNodeRequest peek = lodQueues[lod].peek();
                    double dist = estimateDistance(peek);
                    if (dist < bestDist || (dist == bestDist && lod < bestLod)) {
                        best = peek;
                        bestLod = lod;
                        bestDist = dist;
                    }
                }
            }
            
            // Dequeue the best candidate
            if (best != null) {
                lodQueues[bestLod].poll();
                RequestKey key = RequestKey.of(best);
                queuedKeys.remove(key);
                inFlightKeys.add(key);
                return best;
            }
            
            return null;
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Dequeue from a specific LOD level.
     * 
     * @param lod LOD level [0, 4]
     * @return Next request at that LOD, or null
     */
    public static VoxyRequestDecoder.VoxyNodeRequest dequeue(int lod) {
        if (lod < 0 || lod > 4) {
            return null;
        }
        
        lock.writeLock().lock();
        try {
            VoxyRequestDecoder.VoxyNodeRequest req = lodQueues[lod].poll();
            if (req != null) {
                RequestKey key = RequestKey.of(req);
                queuedKeys.remove(key);
                inFlightKeys.add(key);
            }
            return req;
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Mark a request as finished so it can be enqueued again in the future.
     */
    public static void markCompleted(VoxyRequestDecoder.VoxyNodeRequest request) {
        if (request == null) {
            return;
        }
        lock.writeLock().lock();
        try {
            inFlightKeys.remove(RequestKey.of(request));
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Mark a request as not yet done and re-queue it if it is still unique.
     */
    public static void requeue(VoxyRequestDecoder.VoxyNodeRequest request) {
        if (request == null || request.lodLevel < 0 || request.lodLevel > 4) {
            return;
        }

        lock.writeLock().lock();
        try {
            RequestKey key = RequestKey.of(request);
            inFlightKeys.remove(key);
            if (queuedKeys.contains(key)) {
                return;
            }
            lodQueues[request.lodLevel].offer(request);
            queuedKeys.add(key);
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Get total queue size across all LODs.
     */
    public static int size() {
        lock.readLock().lock();
        try {
            int total = 0;
            for (Queue<?> queue : lodQueues) {
                total += queue.size();
            }
            return total;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get queue size for a specific LOD.
     */
    public static int sizeForLod(int lod) {
        if (lod < 0 || lod > 4) {
            return 0;
        }
        lock.readLock().lock();
        try {
            return lodQueues[lod].size();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Clear all queues.
     */
    public static void clear() {
        lock.writeLock().lock();
        try {
            for (Queue<?> queue : lodQueues) {
                queue.clear();
            }
            queuedKeys.clear();
            inFlightKeys.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** Number of requests currently being processed by a consumer. */
    public static int inFlightSize() {
        lock.readLock().lock();
        try {
            return inFlightKeys.size();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Check if any queue has pending requests.
     */
    public static boolean hasWork() {
        lock.readLock().lock();
        try {
            for (Queue<?> queue : lodQueues) {
                if (!queue.isEmpty()) {
                    return true;
                }
            }
            return false;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Update player position in section-space units for distance estimation.
     */
    public static void updatePlayerSection(int sectionX, int sectionZ) {
        playerSectionX = sectionX;
        playerSectionZ = sectionZ;
    }

    /**
     * Estimate distance from request to player (simplified).
     * 
     * Uses player-relative section-space distance and a mild LOD penalty so
     * nearby requests are preferred while still allowing coarser levels to drain.
     */
    private static double estimateDistance(VoxyRequestDecoder.VoxyNodeRequest req) {
        if (req == null) {
            return Double.MAX_VALUE;
        }
        // Squared distance ignoring Y (vertical), scaled by LOD
        double dx = req.worldX - playerSectionX;
        double dz = req.worldZ - playerSectionZ;
        double distSq = dx * dx + dz * dz;
        
        // Scale by LOD: higher LOD = slightly less urgent
        double lodScale = 1.0 + (req.lodLevel * 0.1);
        return Math.sqrt(distSq) * lodScale;
    }
}
