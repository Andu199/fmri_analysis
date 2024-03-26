import os

if __name__ == "__main__":
    data_dir = "../data/ds000030_R1.0.4/derivatives/fmriprep"
    for dirname in os.listdir(data_dir):
        dir_anat = os.path.join(data_dir, dirname, "anat")
        for filename in os.listdir(dir_anat):
            if "MNI152NLin2009cAsym" not in filename:
                os.remove(os.path.join(dir_anat, filename))
                continue
            if "_brainmask" not in filename and "_preproc" not in filename:
                os.remove(os.path.join(dir_anat, filename))
                continue

        dir_func = os.path.join(data_dir, dirname, "func")
        for filename in os.listdir(dir_func):
            if "rest" not in filename:
                os.remove(os.path.join(dir_func, filename))
                continue
            if "confounds" not in filename and "MNI152NLin2009cAsym" not in filename:
                os.remove(os.path.join(dir_func, filename))
                continue
