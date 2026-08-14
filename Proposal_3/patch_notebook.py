import json

with open("Proposal_3.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        
        # Replace RestorationPipeline_E5
        source = [line.replace("RestorationPipeline_E5", "Restoration_Pipeline_P3") for line in source]
        
        # Replace train_test_split and add synthetic data
        new_source = []
        for line in source:
            new_source.append(line)
            if line.strip() == "test_size=cfg.val_split, random_state=cfg.seed,":
                pass # just keep adding
            if line == "# ── Build dataloaders ───────────────────────────────────────────────────\n":
                if "# ---- Add Synthetic Dataset to Train ONLY ----\n" not in new_source:
                    idx = new_source.index(line)
                    # We insert before this line
                    injection = [
                        "# ---- Add Synthetic Dataset to Train ONLY ----\n",
                        "synth_noisy = sorted(Path(\"data/synthetic/NoisyLR\").glob(\"*.npy\"))\n",
                        "synth_gt = sorted(Path(\"data/synthetic/GT\").glob(\"*.npy\"))\n",
                        "if len(synth_noisy) > 0:\n",
                        "    train_noisy.extend(synth_noisy)\n",
                        "    train_gt.extend(synth_gt)\n",
                        "    print(f\"Added {len(synth_noisy)} synthetic pairs to training.\")\n",
                        "# ---------------------------------------------\n",
                        "\n"
                    ]
                    new_source = new_source[:-1] + injection + [line]
        source = new_source
        
        # Add model.train() and model.eval()
        new_source = []
        for line in source:
            if line == "    epoch_train_losses = []\n" and "    # ── Train" in "".join(source):
                # Check if it's the right place
                new_source.append("    model.train()\n")
            if line == "    epoch_val_losses = []\n" and "    # ── Validate" in "".join(source):
                new_source.append("    model.eval()\n")
            new_source.append(line)
        source = new_source
                
        cell["source"] = source

with open("Proposal_3.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
