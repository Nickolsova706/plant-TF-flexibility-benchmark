import os
from Bio import SeqIO

# --- PATHS ---
# Where your "dirty" files are now
INPUT_BASE = "/home/debian/msc_dissert/Arabidopsis_project/loop_ml_training/06_extracted_fasta/"
# The NEW folder for clean data
OUTPUT_BASE = "/home/debian/msc_dissert/Arabidopsis_project/loop_ml_training/keep_just_atgc_fasta/"

VALID_BASES = set("ATGCatgc")

# Create the main output folder if it doesn't exist
if not os.path.exists(OUTPUT_BASE):
    os.makedirs(OUTPUT_BASE)
    print(f"📁 Created main folder: {OUTPUT_BASE}")

def filter_and_copy(tf_name):
    input_folder = os.path.join(INPUT_BASE, tf_name)
    output_folder = os.path.join(OUTPUT_BASE, tf_name)
    
    # Create the TF subfolder inside the clean directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Files to look for
    files = [
        (f"{tf_name}_positives.fasta", "positives"),
        (f"{tf_name}_negatives.fasta", "negatives")
    ]

    for filename, label in files:
        in_path = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, filename) # Keep same name, different folder

        if not os.path.exists(in_path):
            continue

        original_recs = list(SeqIO.parse(in_path, "fasta"))
        # FILTER: Keep only pure ATGC
        filtered_recs = [r for r in original_recs if all(b in VALID_BASES for b in str(r.seq))]
        
        # Save to the NEW location
        SeqIO.write(filtered_recs, out_path, "fasta")
        
        deleted = len(original_recs) - len(filtered_recs)
        print(f"    - {label}: Kept {len(filtered_recs)} | Removed {deleted} peaks.")

# --- MAIN LOOP ---
print(f"🧹 Starting cleaning process...")

# Get all TF folders
tf_folders = [d for d in os.listdir(INPUT_BASE) if os.path.isdir(os.path.join(INPUT_BASE, d))]

for tf in sorted(tf_folders):
    print(f"\nProcessing TF: {tf}")
    filter_and_copy(tf)

print(f"\n✅ Done! All clean files are now in: {OUTPUT_BASE}")
