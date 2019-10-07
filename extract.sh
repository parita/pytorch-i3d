python extract_features.py \
  -mode rgb \
  -root /proj/vondrick/datasets/TVQA/videos/frames_hq/ \
  -load_model models/rgb_charades.pt \
  -save_dir ./tvqa-rgb-features \
  -gpu 5
