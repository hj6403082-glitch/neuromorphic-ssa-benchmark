import glob
import h5py

files = glob.glob("data/ebssa/*.h5")
if not files:
    print("No .h5 file found in data/ebssa/. Download it first and re-run.")
    raise SystemExit(1)

h5_path = files[0]
print(f"Inspecting: {h5_path}\n")

with h5py.File(h5_path, "r") as f:
    top_keys = list(f.keys())
    print(f"Top-level keys ({len(top_keys)} total): {top_keys[:10]}")
    print("\nStructure of first entry:")
    first = f[top_keys[0]]
    if isinstance(first, h5py.Group):
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name} | shape={obj.shape} | dtype={obj.dtype}")
            else:
                print(f"  Group  : {name}/")
        first.visititems(show)
    else:
        print(f"  Dataset: {top_keys[0]} | shape={first.shape} | dtype={first.dtype}")
