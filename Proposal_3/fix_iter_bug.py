import json

with open("Proposal_3.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        
        # Check if this is the training loop cell
        if any("for epoch in range(cfg.num_epochs):\n" in line for line in source):
            # We want to move train_iter = iter(train_loader) and val_iter = iter(val_loader) inside the loop
            new_source = []
            has_train_iter = False
            has_val_iter = False
            for line in source:
                if line == "train_iter = iter(train_loader)\n":
                    has_train_iter = True
                    continue
                if line == "val_iter   = iter(val_loader)\n":
                    has_val_iter = True
                    continue
                if line == "for epoch in range(cfg.num_epochs):\n":
                    new_source.append(line)
                    if has_train_iter:
                        new_source.append("    train_iter = iter(train_loader)\n")
                    if has_val_iter:
                        new_source.append("    val_iter   = iter(val_loader)\n")
                    continue
                
                new_source.append(line)
            cell["source"] = new_source

with open("Proposal_3.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
