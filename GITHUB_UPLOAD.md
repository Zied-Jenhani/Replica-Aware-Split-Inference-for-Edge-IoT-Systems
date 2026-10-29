# Initial upload

Extract the ZIP and open a terminal inside the extracted
`Replica-Aware-Split-Inference-for-Edge-IoT-Systems` folder, where `README.md`
and `solve_milp.py` are visible. These commands are for that new folder and the
empty repository shown in the screenshot.

```bash
git init -b main
git add .
git status --short
git commit -m "Add simulation scripts, input datasets, and reproduction instructions"
git remote add origin https://github.com/Zied-Jenhani/Replica-Aware-Split-Inference-for-Edge-IoT-Systems.git
git push -u origin main
```

GitHub authentication must be configured on the computer used to push. If Git
asks for an author identity or authentication, resolve that prompt before
retrying the commit or push. These steps do not require uploading the ZIP itself:
the extracted scripts and datasets become browsable files in the repository.

Refresh the repository page after the push. Its root should display the README,
five Python scripts, requirements, citation, and input folders.

For later changes:

```bash
git status --short
git add PATH_TO_CHANGED_FILE
git commit -m "Describe the change"
git push
```

Create a publication release only after confirming that the committed scripts,
inputs, and settings correspond to the reported results.
