# Continuous max-flow augmentation of self-supervised few-shot learning on SPECT left ventricles
## Supporting code for the paper

1. Pre-trained model, named jps_more_encoding_blocks_dropout2_target_task11.pth can be loaded from: https://ikelte-my.sharepoint.com/:u:/g/personal/szaqaei_inf_elte_hu/ESpjLT-Jh5lLjZpJrnjutgwBRKZzJGPtzVDFmVc2W-_THQ?e=AJXCUH it must be placed in the folder nbs/saved_models
2. src/ contains algorithms for CMF, Forward-Back Projection; algs/ contain the shape prior generation code
3. Solution algorithm is in nbs/self_supervised_lv_seg/trained_ssl_3d_unet_cmf.ipnyb