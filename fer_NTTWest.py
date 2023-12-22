# FERライブラリを用いてNTT西日本のCMから顔を抽出し、その感情を分析する
# 参考資料：https://qiita.com/MMsk0914/items/838a7685a984461c8cb8

import cv2
from fer import FER


INPUT_PATH = "../data/NTTWest.mp4"
OUTPUT_PATH = "../data/NTTWest_FER.mp4"

# 顔を囲う線の太さ
THICKNESS = 2


def Main():

    fer = FER(mtcnn=True)

    # OpenCVの動画関連：https://www.learning-nao.com/?p=1582
    cap = cv2.VideoCapture(INPUT_PATH)
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc("m", "p", "4", "v"), cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # 色（BGR）
    # angry = red, disgust = purple, fear = cyan, happy = pink, sad = blue,
    # surprise = orange, neutral = green
    color_dict = {"angry": (0, 0, 255), 
                "disgust": (128, 0, 128),
                "fear": (255, 255, 0),
                "happy": (255, 0, 255),
                "sad": (255, 0, 0),
                "surprise": (0, 165, 255),
                "neutral": (0, 255, 0)}
    
    #i = 1
    while True:
        #print("Frame: " + str(i))
        ret, img = cap.read()
        if not ret or cv2.waitKey(1) & 0xFF == ord("q"):
            break


        # 顔検出・感情スコア算出
        captured_emotions = fer.detect_emotions(img)
        #print(captured_emotions)

        # 画像内で最も表している感情を表示
        #dominant_emotion, emotion_score = fer.top_emotion(img)


        # 顔検出数分ループ
        for idx, emo in enumerate(captured_emotions):

            # その顔が最も表している感情を抽出
            emotion = emo["emotions"]
            top_emotion = max(emotion, key=emotion.get)

            # (左上x座標, 左上y座標, xの長さ, yの長さ)   e.g. (100, 50, 10, 10) 
            box = emo["box"]
            img = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), 
                                color_dict[top_emotion], thickness=THICKNESS)
                        
        writer.write(img)
        cv2.imshow("Video", img)
        #i += 1


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Main()
