import numpy as np

# librosa와 호환되지 않는 np.complex 문제를 임시로 해결
if not hasattr(np, 'complex'):
    np.complex = complex

import librosa

# 1. 오디오 로드 및 비트 분석
def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return tempo, beat_times

# 2. 노트 배치 (간단한 밀도 기반 배치)
def generate_notes(beat_times, density=2):
    notes = []
    for i, beat in enumerate(beat_times):
        if i % density == 0:
            notes.append(beat)
    return notes

# 3. .bms 파일 출력
def write_bms(notes, bpm, output_path, wav_id="01"):
    with open(output_path, "w") as f:
        f.write("#PLAYER 1\n")
        f.write("#TITLE Auto Metal Chart\n")
        f.write(f"#BPM {int(bpm)}\n")
        f.write(f"#WAV{wav_id} song.ogg\n")
        f.write("#GENRE Metal\n")
        f.write("#RANK 2\n")
        f.write("#PLAYLEVEL 5\n")
        f.write("#TOTAL 200\n")
        f.write("#STAGEFILE stage.png\n")
        f.write("#ARTIST AutoGenerator\n")
        f.write("#DIFFICULTY HARD\n")
        f.write("\n")

        for idx, beat_time in enumerate(notes):
            measure = int(idx / 4) + 1
            channel = "11"
            f.write(f"#{measure:03d}{channel}:01\n")

# 테스트 실행
audio_path = "/mnt/data/sample.ogg"  # 여기에 .ogg 파일 경로를 넣어야 합니다
output_bms_path = "/mnt/data/auto_chart.bms"

try:
    bpm, beat_times = analyze_audio(audio_path)
    notes = generate_notes(beat_times, density=2)
    write_bms(notes, bpm, output_bms_path)
    result = f".bms 파일이 성공적으로 생성되었습니다: {output_bms_path}"
except Exception as e:
    result = f"오류 발생: {e}"

result
