import sys; sys.path.insert(0, 'python')
import cudtw
import numpy as np

# Initialize
cudtw.init(0)

# Load query
query = cudtw.load_bin('target/395.bin')  # 395 floats

# Process entire dataset
cudtw.set_query(query)
stats = cudtw.process_folder(
    sequences_dir='sequences/',
    result_dir='result/',
    verbose=True
)

print(f"Done: {stats['sequences']} sequences in {stats['wall_seconds']:.1f}s")
