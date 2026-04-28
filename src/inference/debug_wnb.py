import wandb
import base64
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

WANDB_ENTITY = os.getenv("WANDB_ENTITY") or ""
WANDB_PROJECT = os.getenv("WANDB_PROJECT") or ""
WANDB_RUN_ID = os.getenv("WANDB_RUN_ID") or ""

# artifact = api.artifact("myuser/myproject/model:v3")
# artifact_dir = artifact.download()

api = wandb.Api()
run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RUN_ID}")

# print(f"Run ID: {run.id}\nRun name is: {run.name}")

# for artifact in run.logged_artifacts():
#     print(f"Artifact name: {artifact.name}")
#     print(f"Artifact version: {artifact.version}")
#     print(f"Artifact created_at: {artifact.created_at}")
#     print(f"Artifact type: {artifact.type}")
#     # print(f"Artifact aliases: {artifact.aliases}") # e.g.: latest (default)


# # --- load in given n of runs: ----
# runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", per_page=5)
# for r in runs:
#     print(r.id)
#     print(r.name)
#     print(r.created_at)

# fdf = runs.summary
# print(type(fdf))
# print(fdf)

# # # --- getting history ----

# keys = ["epoch", "train_loss", "val_loss"]
# sys_keys = ["system.cpu", "system.memory", "system.disk"]

# df = run.history(keys=sys_keys, samples=1000)
# df.to_dict(orient="records")
# df = pd.DataFrame(df)
# print(df)


## ---- LOAD TRAIN METRICS -----
# history = run.history()

# train_loss = history["train_loss"]
# val_loss = history["val_loss"]
# test_acc = history["test_accuracy"]
# print(train_loss[:10])


## ---- LOAD IMAGES -----
# run.log({"test_confusion_matrix": wandb.Image(fig)})
# run.log({"test_clf_report": wandb.Html(...)})

## Files (images, html)

# for f in run.files():
#     if any(x in f.name for x in ["confusion_matrix", "clf_report"]):
#         f.download(exist_ok=True)

## by default saves it to media/images/test_confusion_matrix_10_c32c5b8ecafbbe85a001.png
## and: media/html/test_clf_report_9_baffe2b9cd9a92f4cf04.html


#  ----- LOAD ARTIFACT ------
# artifact = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run.name}:v0")
# artifact_dir = artifact.download()

## by default saves it to artifacts/ft-top...pt



## ---- SYSTEM METRICS -----
## Would be nice, but I could not find any, even though the wandb webapp UI shows them logged under 'Charts'
#import json 

# files = run.files()
# for f in files:
#     if "wandb-summary" in f.name:
#         path = f.download(exist_ok=True).name

#         with open(path) as fp:
#             file = fp.read()

#         print(f"HERE IS {f.name}")
#         print(file)

# scan = run.scan_history()

# with open("w_ins.txt", "w") as wf:
#     text = " ".join([f"{str(k)}" for k in scan])
#     wf.write(text)

