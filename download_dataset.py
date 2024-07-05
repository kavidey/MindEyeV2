from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download(repo_id="pscotti/mindeyev2", repo_type = "dataset", revision="main", allow_patterns="*.tar",
    local_dir= "/mnt/isilon/CSC6/HelenZhouLab/HZLHD0/InternsnStudents/Interns/kavi/data/datasets/nsd-mind-eye", local_dir_use_symlinks = False, resume_download = True)
snapshot_download(repo_id="pscotti/mindeyev2", repo_type = "dataset", revision="main", allow_patterns="*.hdf5",
    local_dir= "/mnt/isilon/CSC6/HelenZhouLab/HZLHD0/InternsnStudents/Interns/kavi/data/datasets/nsd-mind-eye", local_dir_use_symlinks = False, resume_download = True,
    max_workers = 2)
# hf_hub_download(repo_id="pscotti/mindeyev2", filename="coco_images_224_float16.hdf5", repo_type="dataset")
hf_hub_download(repo_id="pscotti/mindeyev2", filename="sd_image_var_autoenc.pth", repo_type="dataset",
    local_dir= "/mnt/isilon/CSC6/HelenZhouLab/HZLHD0/InternsnStudents/Interns/kavi/data/datasets/nsd-mind-eye", local_dir_use_symlinks = False, resume_download = True)
hf_hub_download(repo_id="pscotti/mindeyev2", filename="train_logs/final_subj01_pretrained_40sess_24bs/last.pth", repo_type="dataset",
    local_dir= "/mnt/isilon/CSC6/HelenZhouLab/HZLHD0/InternsnStudents/Interns/kavi/data/datasets/nsd-mind-eye", local_dir_use_symlinks = False, resume_download = True)
hf_hub_download(repo_id="pscotti/mindeyev2", filename="bigG_to_L_epoch8.pth", repo_type="dataset",
    local_dir= "/mnt/isilon/CSC6/HelenZhouLab/HZLHD0/InternsnStudents/Interns/kavi/data/datasets/nsd-mind-eye", local_dir_use_symlinks = False, resume_download = True)
hf_hub_download(repo_id="pscotti/mindeyev2", filename="unclip6_epoch0_step110000.ckpt", repo_type="dataset",
    local_dir= "/mnt/isilon/CSC6/HelenZhouLab/HZLHD0/InternsnStudents/Interns/kavi/data/datasets/nsd-mind-eye", local_dir_use_symlinks = False, resume_download = True)
hf_hub_download(repo_id="pscotti/mindeyev2", filename="zavychromaxl_v30.safetensors", repo_type="dataset",
    local_dir= "/mnt/isilon/CSC6/HelenZhouLab/HZLHD0/InternsnStudents/Interns/kavi/data/datasets/nsd-mind-eye", local_dir_use_symlinks = False, resume_download = True)
hf_hub_download(repo_id="pscotti/mindeyev2", filename="gnet_multisubject.pt", repo_type="dataset",
    local_dir= "/mnt/isilon/CSC6/HelenZhouLab/HZLHD0/InternsnStudents/Interns/kavi/data/datasets/nsd-mind-eye", local_dir_use_symlinks = False, resume_download = True)