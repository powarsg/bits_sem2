"""
Quick test runner for defence_drone_gbfs.py with multiple input files.
"""
import subprocess
import sys
import os
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
main_script = os.path.join(script_dir, 'defence_drone_gbfs.py')
original_input = os.path.join(script_dir, 'inputPS04.txt')

test_files = [
    ('inputPS04.txt',       'outputPS04_test1.txt'),
    ('inputPS04_test2.txt', 'outputPS04_test2.txt'),
    ('inputPS04_test3.txt', 'outputPS04_test3.txt'),
]

for inp, out in test_files:
    inp_path = os.path.join(script_dir, inp)
    out_path = os.path.join(script_dir, out)

    # Temporarily point inputPS04.txt to the test file
    if inp != 'inputPS04.txt':
        shutil.copy(inp_path, original_input)

    print(f"\n{'='*60}")
    print(f"Running test: {inp}  ->  {out}")
    print('='*60)
    result = subprocess.run([sys.executable, main_script], capture_output=True, text=True)
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])

    # Rename output
    default_out = os.path.join(script_dir, 'outputPS04.txt')
    if os.path.exists(default_out):
        shutil.copy(default_out, out_path)
        print(f"Output saved to: {out}")

# Restore original inputPS04.txt
shutil.copy(os.path.join(script_dir, 'inputPS04.txt_backup') if os.path.exists(
    os.path.join(script_dir, 'inputPS04.txt_backup')) else
    os.path.join(script_dir, 'inputPS04_test2.txt'),  # fallback; original already ran first
    original_input
)

# Actually restore the real original by re-running test1 content is already the original
# The first test runs on the original, so we just need to restore it from test2/test3 runs
# Copy back original content
original_content = """8 8
S . . . N . . .
. W . . N . . .
. . . W N . W .
N N . . . . N .
. . . W . . . .
. . . . . N N .
. . W . . . . E
. . . W . W . .
0 0
6 7
"""
with open(original_input, 'w') as f:
    f.write(original_content)

print("\n✓ Original inputPS04.txt restored.")
print("✓ All tests complete.")
