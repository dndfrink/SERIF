import time
import random
import string
import traceback
from enum import Enum
from typing import Dict
import os
from src.kvcache import KVCache
from multiprocessing import shared_memory, Process

class Operation(Enum):
    SET = 1
    GET = 2
    DELETE = 3

def createRandomBytes(size: int) -> bytes:
    return random.randbytes(size)

def createRandomKey(length: int = 30) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def processWorker(modality_map: Dict[str, Dict[str, int]], testData: Dict[str, list], numOperations: int, processId: int):
    try:
        print(f"Process {processId} (PID {os.getpid()}) starting...")
        cache = KVCache(modality_map=modality_map)
        
        # Track which keys are currently set in the cache
        activeKeys = {modality: set() for modality in modality_map}
        
        for i in range(numOperations):
            # Pick random modality
            modality = random.choice(list(modality_map.keys()))
            # Pick random operation
            operation = random.choice(list(Operation))
            
            try:
                if operation == Operation.SET:
                    keyValueIdx = random.randrange(len(testData[modality]))
                    key, value = testData[modality][keyValueIdx]
                    if cache.set(modality, key, value):
                        activeKeys[modality].add(key)
                    print(f"Process {processId} - Op {i+1}: SET {modality}[{key}]")
                elif operation == Operation.GET:
                    if activeKeys[modality]:
                        key = random.choice(list(activeKeys[modality]))
                    else:
                        key, _ = random.choice(testData[modality])
                    value = cache.get(modality, key)
                    result = "found" if value is not None else "not found"
                    print(f"Process {processId} - Op {i+1}: GET {modality}[{key}] -> {result}")
                elif operation == Operation.DELETE:
                    if activeKeys[modality]:
                        key = random.choice(list(activeKeys[modality]))
                        success = cache.delete(modality, key)
                        if success:
                            activeKeys[modality].remove(key)
                    else:
                        key, _ = random.choice(testData[modality])
                        success = cache.delete(modality, key)
                    print(f"Process {processId} - Op {i+1}: DELETE {modality}[{key}] -> {'success' if success else 'not found'}")
            except Exception as e:
                print(f"\nProcess {processId} - Operation {i+1} failed:")
                print(f"Operation: {operation.name}")
                print(f"Modality: {modality}")
                print(traceback.format_exc())
                return
        print(f"Process {processId} completed successfully")
        
    except Exception as e:
        print(f"Process {processId} failed with error:")
        print(traceback.format_exc())

def testKvCache(modality_map: Dict[str, Dict[str, int]], numProcesses: int = 5, numOperations: int = 100):
    print("\nStarting Multi-Process Test:")
    print("Modality Map Configuration:")
    for modality, settings in modality_map.items():
        print(f"  {modality}: {settings}")
    
    totalMemory = sum((m["size"] + 8) * m["entries"] for m in modality_map.values())
    print(f"\nTotal Memory Required: {totalMemory / (1024**3):.2f} GB")

    # Initialize the cache first in the parent process
    try:
        print("\nInitializing KVCache in parent process...")
        start_time = time.time()
        cache = KVCache(modality_map)
        print(f"Initialization successful: {time.time() - start_time:.3f} seconds")
    except Exception as e:
        print("Failed during initialization:")
        print(traceback.format_exc())
        cleanupSharedMemory()
        return

    # Prepare test data
    testData = {}
    for modality, config in modality_map.items():
        print(f"\nPreparing test data for modality: {modality}")
        testData[modality] = [
            (createRandomKey(), createRandomBytes(config["size"]))
            for _ in range(config["entries"])
        ]

    # Launch worker processes
    processes = []
    for i in range(numProcesses):
        p = Process(target=processWorker, 
                   args=(modality_map, testData, numOperations, i))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("\nAll processes completed")
    cleanupSharedMemory()

def cleanupSharedMemory():
    try:
        shared_memory.SharedMemory(name='kvcache').unlink()
        shared_memory.SharedMemory(name='kvcache_metadata').unlink()
    except Exception:
        pass

if __name__ == "__main__":
    # Test configurations
    testConfigs = [
        # Start with a small test configuration
        {
            "smallData": {"size": 1024, "entries": 10},    # 10 entries of 1KB each
        },
        # Medium config
        {
            "mediumData": {"size": 50000000, "entries": 20},  # 20 entries of 10KB each
            "smallData": {"size": 1024, "entries": 40},    # 40 entries of 1KB each
        }
    ]

    # Run tests
    for i, config in enumerate(testConfigs, 1):
        print(f"\n{'='*50}")
        print(f"Running Test Configuration {i}")
        print('='*50)
        testKvCache(config, numProcesses=15, numOperations=2000)