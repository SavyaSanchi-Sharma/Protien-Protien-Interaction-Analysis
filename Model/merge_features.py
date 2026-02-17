import pandas as pd
import os

def merge_struct_pssm(struct_csv, pssm_csv, output_csv):
    print(f"\n{'='*80}")
    print(f"[MERGE] {os.path.basename(struct_csv)} + {os.path.basename(pssm_csv)}")
    print(f"{'='*80}\n")
    print("[1/4] Loading structural features...")
    df_struct = pd.read_csv(struct_csv)
    print(f"      Shape: {df_struct.shape}")

    print("[2/4] Loading PSSM features...")
    df_pssm = pd.read_csv(pssm_csv)
    print(f"      Shape: {df_pssm.shape}")
    df_struct['PDB']=df_struct['PDB'].str.lower()
    df_pssm['PDB']=df_pssm['PDB'].str.lower()
    print("[2.5/4] Filtering PSSM to PDBs present in 16D...")
    valid_pdbs = set(df_struct['PDB'].unique())
    df_pssm = df_pssm[df_pssm['PDB'].isin(valid_pdbs)]
    print(f"      Shape after PDB filter: {df_pssm.shape}")

    print("[3/4] Merging on (PDB, Chain, ResIdx)...")
    df_merged = pd.merge(
        df_struct,
        df_pssm,
        on=['PDB', 'Chain', 'ResIdx', 'AA', 'Label'],
        how='inner'
    )
    print(f"      Merged shape: {df_merged.shape}")
    missing = len(df_struct) - len(df_merged)
    if missing > 0:
        print(f"      [WARN] {missing} residues lost in merge ({missing/len(df_struct)*100:.1f}%)")
    expected_cols = (
        ['PDB', 'Chain', 'ResIdx', 'AA', 'Label'] +  # 5 metadata
        ['RSA', 'ResFlex', 'Hydrophobicity', 'PackingDensity', 
         'HSE_up', 'HSE_down',
         'Poly_RSA', 'Poly_Flex', 'Poly_interaction', 'BondAngle'] +  # 10 scalar
        [f'PSSM_{i}' for i in range(1, 21)] +  # 20 PSSM
        ['sin_phi', 'cos_phi', 'sin_psi', 'cos_psi', 'sin_omega', 'cos_omega']  # 6 angles
    )
    df_merged = df_merged[expected_cols]
    print("[4/4] Saving merged features...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    print(f"      Output: {output_csv}")
    print(f"\n{'='*80}")
    print(f"[SUMMARY]")
    print(f"{'='*80}")
    print(f"Total residues: {len(df_merged)}")
    print(f"Total features: {len(df_merged.columns) - 5}  (5 metadata + 36 features)")
    print(f"Breakdown:")
    print(f"  - Scalar features: 10D")
    print(f"  - PSSM features:   20D")
    print(f"  - Angle features:   6D")
    print(f"  - TOTAL:           36D")
    print(f"{'='*80}\n")
    
    return df_merged


if __name__ == "__main__":
    os.makedirs("data/features", exist_ok=True)
    datasets = [
        ("Train_335", "Train_335_16D.csv", "Train_335_pssm.csv", "Train_335_36D.csv"),
        ("Test_60", "Test_60_16D.csv", "Test_60_pssm.csv", "Test_60_36D.csv"),
        ("Test_315", "Test_315_16D.csv", "Test_315_pssm.csv", "Test_315_36D.csv"),
    ]
    for name, struct_file, pssm_file, output_file in datasets:
        merge_struct_pssm(
            f"data/structural/{struct_file}",
            f"data/pssm/{pssm_file}",
            f"data/features/{output_file}"
        )
    print("[COMPLETE] All datasets merged successfully! 🎉")
    print("\nNext steps:")
    print("1. Update createdatset.py to load from data/features/*_36D.csv")
    print("2. Delete old normalization: rm data/struct_norm.npz")
    print("3. Run training: python train.py")
