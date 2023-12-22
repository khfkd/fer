# 感情分析FER（Face Emotion Recognizer）の実装
映像を用いたFERを実装  
（今回はNTT西日本のCM映像を用いて実装しましたが、ファイルはgithub上に置いていません）

- 使い方  
1. パッケージインストール
```bash
conda install -c conda-forge face_recognition
pip install FER
pip install tensorflow==2.13.0
```

2. ファイル設定
- 動画（mp4形式）を任意のフォルダに格納
- **fer_NTTWest.py**の入力動画パス（INPUT_PATH）を使用動画に対応させる

3. 実行
```bash
python fer_NTTWest.py
```
