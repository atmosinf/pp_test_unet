# pp_test_unet

This repository contains the implementation for a U-Net model.

## 🧠 Model Hosting

Due to file size limits on GitHub, trained models are hosted on **Hugging Face Hub**.

### Latest Model
- **Repository**: [amoelgeogy/pp-test-unet](https://huggingface.co/amoelgeogy/pp-test-unet)
- **Latest Upload**: [pp-test-unet_unet_20260511_170317.pth](https://huggingface.co/amoelgeogy/pp-test-unet/blob/main/pp-test-unet_unet_20260511_170317.pth)

### How to Upload New Models
We use a custom script `hf_upload.py` to upload models with the following naming convention: `repo_name_model_name_datetime.extension`.

To upload a new model:
```bash
source venv/bin/activate
python hf_upload.py <path_to_model> amoelgeogy/pp-test-unet
```