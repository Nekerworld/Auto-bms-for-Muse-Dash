import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from collections import deque
import librosa
import time
import os
import zipfile
import tempfile
import shutil

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_callback=None, total_epochs=1):
        super().__init__()
        self.progress_callback = progress_callback
        self.start_time = None
        self.last_time_update = 0
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.epoch_start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.last_time_update = self.start_time
        # 파일 단위 진행률은 상위에서 관리

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.epoch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        now = time.time()
        elapsed_time_total = now - self.start_time
        elapsed_time_epoch = now - self.epoch_start_time

        # 배치 손실 및 경과 시간 업데이트 (매 1초 또는 배치 끝마다)
        if now - self.last_time_update >= 0.1 or batch == 0 or logs.get('loss') is not None:
             data = {
                 'progress_type': 'batch',
                 'epoch': self.current_epoch,
                 'total_epochs': self.total_epochs, # 전체 에폭 수 추가
                 'loss': float(logs.get('loss', 0)),
                 'elapsed_time_total': elapsed_time_total,
                 'elapsed_time_epoch': elapsed_time_epoch, # 에폭 경과 시간 추가
             }
             if hasattr(self.progress_callback, 'emit'):
                 self.progress_callback.emit(data)
             elif self.progress_callback:
                 self.progress_callback(data)
             self.last_time_update = now

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_callback:
            try:
                elapsed_time_total = time.time() - self.start_time
                elapsed_time_epoch = time.time() - self.epoch_start_time
                data = {
                    'progress_type': 'epoch_end',
                    'epoch': epoch + 1,
                    'total_epochs': self.total_epochs, # 전체 에폭 수 추가
                    'status': '에폭 완료', # 이 상태는 이제 필요 없을 수 있지만 유지
                    'loss': float(logs.get('loss', 0)),
                    'val_loss': float(logs.get('val_loss', 0)),
                    'elapsed_time_total': elapsed_time_total,
                    'elapsed_time_epoch': elapsed_time_epoch, # 에폭 경과 시간 추가
                }
                if hasattr(self.progress_callback, 'emit'):
                    self.progress_callback.emit(data)
                elif self.progress_callback:
                    self.progress_callback(data)
            except Exception as e:
                print(f"Warning: Error in progress callback: {str(e)}")

# 1. 오디오 전처리 모듈
class AudioPreprocessor:
    def __init__(self, sr=22050, n_mels=128):
        self.sr = sr
        self.n_mels = n_mels

    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            # 오디오 길이가 0인 경우 처리
            if len(y) == 0:
                print(f"Warning: Audio file {audio_path} is empty or cannot be loaded.")
                return None
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            log_mel = librosa.power_to_db(mel)
            return log_mel
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            return None

# BMS 파싱 및 타겟 시퀀스 생성 함수
def normalize_note_type(note_type_str):
    """BMS 노트 타입을 정규화 (16진수 또는 특수 문자 처리)"""
    try:
        # 16진수 변환 시도
        return int(note_type_str, 16)
    except ValueError:
        # 16진수가 아닌 경우 특수 노트 타입 처리 (예: '0H', '0S')
        # Muse Dash의 특수 노트 타입을 숫자로 매핑하거나 무시할 수 있음
        # 여기서는 0으로 처리하거나 특정 값으로 매핑하는 로직 추가
        # 간단히 '0H'/'0S' 등을 무시하도록 0을 반환 (필요에 따라 확장)
        # Muse Dash 관련 특수 노트 타입 목록에 따라 수정 필요
        muse_dash_special_notes = ['0H', '0S'] # 예시, 실제 Muse Dash 규격 확인 필요
        if note_type_str in muse_dash_special_notes:
            return 0 # 무시하거나 다른 값으로 매핑
        # 그 외 알 수 없는 타입은 오류 방지를 위해 0 반환
        print(f"Warning: Unknown note type encountered: {note_type_str}")
        return 0

def parse_bms_to_targets(bms_path):
    """BMS 파일을 읽어 노트 시퀀스(타겟)으로 변환"""
    notes = []
    current_measure_data = {}
    last_measure_num = -1
    notes_per_measure = 16 # Muse Dash 기본 분할 수 (4/4박자 16분음표)
    valid_channels = [f'{i:02d}' for i in range(11, 17)] + ['18', '53', '54'] # Muse Dash 관련 채널 예시

    try:
        with open(bms_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('*'):
                    continue
                # '#MEASURECHANNEL:DATA' 형식 파싱
                if line.startswith('#') and ':' in line and len(line) >= 7:
                    try:
                        parts = line[1:].split(':', 1)
                        measure_channel = parts[0]
                        data = parts[1]

                        measure_num = int(measure_channel[:3])
                        channel = measure_channel[3:5]

                        # 유효한 채널인지 확인
                        if channel not in valid_channels:
                            continue # 무시

                        # 이전 마디 데이터 처리
                        if measure_num != last_measure_num and last_measure_num != -1:
                            # 마디 데이터가 비어있지 않으면 추가
                            if current_measure_data:
                                notes.append(current_measure_data)
                            current_measure_data = {}

                        last_measure_num = measure_num

                        # 노트 데이터 파싱
                        # 데이터 문자열의 길이가 notes_per_measure * 2와 일치하는지 확인 (선택 사항)
                        # if len(data) != notes_per_measure * 2:
                        #    print(f"Warning: Unexpected data length in {bms_path}, measure {measure_num}, channel {channel}")
                        #    continue # 또는 다른 처리 방식 선택

                        for i in range(0, len(data), 2):
                            if (i // 2) < notes_per_measure: # 마디 범위를 벗어나지 않도록 체크
                                pos = i // 2
                                note_val_str = data[i:i+2] # 노트 값 (예: '01', '00', '0F', '0H')
                                note_val = normalize_note_type(note_val_str) # 16진수 또는 특수 타입 처리

                                if note_val != 0: # '00' 또는 무시된 특수 노트가 아닌 경우
                                    if measure_num not in current_measure_data:
                                        current_measure_data[measure_num] = {}
                                    if pos not in current_measure_data[measure_num]:
                                        current_measure_data[measure_num][pos] = []
                                    current_measure_data[measure_num][pos].append({
                                        'channel': channel,
                                        'type': note_val_str # 원본 문자열 유지
                                    })

                    except ValueError as ve:
                        print(f"Error parsing line in {bms_path}: {line} - {ve}")
                        continue # 파싱 오류 라인 건너뛰기
                    except Exception as ex:
                        print(f"Unexpected error processing line in {bms_path}: {line} - {ex}")
                        continue

        # 파일의 마지막 마디 데이터 추가
        if current_measure_data:
             notes.append(current_measure_data)

        # 타겟 시퀀스 생성
        targets = []
        # notes 리스트는 이제 각 요소가 {마디번호: {위치: [노트정보]}} 형태
        # 모든 마디와 위치를 순회하며 타겟 벡터 생성
        all_measure_nums = sorted(list(set(m for d in notes for m in d.keys()))) # 등장한 모든 마디 번호
        max_measure = all_measure_nums[-1] if all_measure_nums else 0

        for measure_num in range(max_measure + 1): # 0 마디부터 마지막 마디까지
             measure_data = next((d for d in notes if measure_num in d), {measure_num: {}}).get(measure_num, {}) # 해당 마디 데이터 가져오기

             for pos in range(notes_per_measure): # 각 마디의 위치 (0~15)
                # 해당 위치에 노트가 있는지 확인
                if pos in measure_data:
                    # 해당 위치의 첫 번째 노트 정보를 사용 (간단화)
                    note_info = measure_data[pos][0]
                    note_type_str = note_info['type']

                    # 타겟 벡터 생성 로직 (채널 및 타입 기반)
                    # 예시: [일반노트_존재, 홀드노트_존재, 보스노트_존재, 시각효과_존재]
                    target_vec = [
                        1.0 if note_info['channel'] in ['11', '13', '14'] else 0.0, # 일반 키 노트
                        1.0 if note_info['channel'] == '15' else 0.0, # 홀드 노트
                        1.0 if note_info['channel'] == '18' else 0.0, # 보스 효과
                        1.0 if note_info['channel'] in ['53', '54'] else 0.0  # 시각 효과
                        # 필요에 따라 다른 속성 (예: 홀드 길이, 노트 타입 상세 구분) 추가 가능
                    ]
                    targets.append(target_vec)
                else:
                    # 해당 위치에 노트가 없으면 0 벡터 추가
                    targets.append([0.0] * 4) # 타겟 벡터 차원 수에 맞게 수정

        if not targets:
            print(f"Warning: No valid note data found in {bms_path} after parsing.")
            return np.array([])

        print(f"Parsed {len(targets)} note positions from {bms_path}")
        return np.array(targets)
    finally:
        pass

def extract_mdm_pairs(mdm_path, extract_dir):
    """mdm 파일에서 반드시 music.ogg만 bms 파일들의 짝으로 사용하여 (music.ogg, bms) 쌍 리스트를 반환합니다."""
    pairs = []
    music_ogg_found = False
    extracted_bms_paths = []
    try:
        with zipfile.ZipFile(mdm_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # 디렉토리는 건너뛰기
                if file_info.is_dir():
                    continue
                # 파일명 추출
                file_name = os.path.basename(file_info.filename)
                target_path = os.path.join(extract_dir, file_name)

                # music.ogg 추출 확인
                if file_name.lower() == 'music.ogg':
                    zip_ref.extract(file_info, extract_dir)
                    music_ogg_path = target_path
                    music_ogg_found = True
                # .bms 파일 추출 확인
                elif file_name.lower().endswith('.bms'):
                    zip_ref.extract(file_info, extract_dir)
                    extracted_bms_paths.append(target_path)

        # music.ogg 파일이 있고 bms 파일들이 추출된 경우 쌍 생성
        if music_ogg_found and extracted_bms_paths:
            for bms_path in extracted_bms_paths:
                 pairs.append((music_ogg_path, bms_path))
        elif not music_ogg_found:
             print(f"Warning: music.ogg not found in {mdm_path}. Cannot create (ogg, bms) pairs.")
        elif not extracted_bms_paths:
             print(f"Warning: No .bms files found in {mdm_path}. Cannot create (ogg, bms) pairs.")

    except zipfile.BadZipFile:
        print(f"Error: {mdm_path} is not a valid zip file.")
    except FileNotFoundError:
        print(f"Error: {mdm_path} not found.")
    except Exception as e:
        print(f"Error extracting mdm {mdm_path}: {str(e)}")

    return pairs

# 2. 임베딩 모델 (간단한 CNN 기반)
class MusicEmbeddingModel:
    def __init__(self, input_shape=(128, None, 1), embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.model = self._build_model(input_shape, embedding_dim)

    def _build_model(self, input_shape, embedding_dim):
        input_layer = layers.Input(shape=(input_shape[0], None, 1))
        x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x) # 여기서 시계열 정보가 압축됨
        x = layers.Dense(embedding_dim, activation='relu')(x) # 임베딩 차원 조정
        return models.Model(inputs=input_layer, outputs=x)

    def extract_embedding(self, features):
        # features shape: (n_mels, T)
        if features is None or features.size == 0: # 오디오 로드 실패 시 처리
             return np.zeros(self.embedding_dim) # 임베딩을 0 벡터로 반환하거나 다른 처리
             
        # 모델 입력 형태 (1, H, W, C) = (1, n_mels, T, 1)
        features = np.expand_dims(features, axis=(0, -1)) 
        
        # 모델 예측 시 오류 방지를 위해 T 차원이 최소 1 이상인지 확인
        if features.shape[2] == 0:
             print("Warning: Feature time dimension is zero, returning zero embedding.")
             return np.zeros(self.embedding_dim)
             
        try:
            embedding = self.model.predict(features)
            return embedding[0] # (embedding_dim,)
        except Exception as e:
            print(f"Error during embedding extraction: {str(e)}")
            return np.zeros(self.embedding_dim) # 오류 발생 시 0 벡터 반환

# 3. 노트 예측 모델 (임베딩→노트 예측만 담당)
class NotePredictionModel:
    def __init__(self, embedding_dim=128, output_dim=4): # 타겟 벡터 차원 수에 맞게 수정
        self.model = self._build_model(embedding_dim, output_dim)

    def _build_model(self, embedding_dim, output_dim):
        model = models.Sequential([
            layers.Input(shape=(embedding_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_dim, activation='sigmoid') # 각 노트 속성의 확률 예측
        ])
        # 다중 라벨 분류처럼 접근 (각 출력이 독립적)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

# 5. 통합 파이프라인 (학습/추론)
class MusicNoteAI:
    def __init__(self, embedding_dim=128, output_dim=4): # 타겟 벡터 차원 수에 맞게 수정
        self.preprocessor = AudioPreprocessor()
        self.embedding_model = MusicEmbeddingModel(embedding_dim=embedding_dim)
        self.note_prediction_model = NotePredictionModel(embedding_dim=embedding_dim, output_dim=output_dim)

    def train(self, mdm_bms_paths, temp_base_dir, epochs=10, progress_callback=None):
        # 데이터 로딩 및 학습을 파일/쌍 단위로 수행
        total_files_or_pairs = 0
        for file_path in mdm_bms_paths:
             if file_path.lower().endswith('.mdm'):
                 # mdm 파일 내의 bms 파일 수만큼 카운트 필요 (대략적으로 1개로 가정하거나 사전 계산)
                 # 정확한 카운트를 위해 extract_mdm_pairs를 미리 실행하거나 zip 내부 확인 필요
                 # 여기서는 간단히 mdm 파일 1개당 1개 이상의 학습 데이터가 나온다고 가정
                 total_files_or_pairs += 1 # 임시 카운트
             elif file_path.lower().endswith('.bms'):
                 total_files_or_pairs += 1 # bms 파일 1개

        processed_count = 0

        for file_path in mdm_bms_paths:
            processed_count += 1
            file_name = os.path.basename(file_path)
            print(f"Processing file {processed_count}/{total_files_or_pairs}: {file_name}")

            pairs_to_process = []
            current_temp_dir = None # 파일별 임시 디렉토리 관리 (main에서 처리)

            if file_path.lower().endswith('.mdm'):
                # MDM 파일 압축 해제 및 (ogg, bms) 쌍 추출
                # 파일별 고유 임시 디렉토리 사용 (main에서 생성하여 전달받는 방식이 더 좋음)
                # 현재는 여기서 임시 디렉토리 생성 및 main의 temp_dirs에 추가하는 방식 시도
                try:
                     # main에서 생성된 temp_base_dir 하위에 파일별 임시 디렉토리 생성
                    current_temp_dir = os.path.join(temp_base_dir, f"extract_{os.path.splitext(file_name)[0]}_{processed_count}")
                    os.makedirs(current_temp_dir, exist_ok=True)
                    # main의 self.temp_dirs에 추가하는 로직이 필요하지만, learning 모듈에서는 직접 접근 어려움
                    # 임시로 여기서 생성 및 정리 시도 또는 main에서 디렉토리 리스트를 전달받아 사용
                    # 여기서는 main에서 temp_base_dir을 받고, 그 하위에 생성하는 방식으로 구현

                    pairs_to_process = extract_mdm_pairs(file_path, current_temp_dir)

                except Exception as e:
                    print(f"Error processing MDM file {file_name}: {str(e)}")
                    if current_temp_dir and os.path.exists(current_temp_dir):
                         shutil.rmtree(current_temp_dir, ignore_errors=True)
                    continue # 다음 파일로 넘어감

            elif file_path.lower().endswith('.bms'):
                # 단일 BMS 파일 처리 (오디오 파일은 어디에 있는지? - 학습 데이터셋 구성 문제)
                # 현재 구조상 BMS 단독 학습은 오디오 특징 추출 때문에 어려움
                # BMS 파일만 있는 경우를 위한 별도 처리 또는 오류 메시지 필요
                # 일단은 BMS 파일만 있는 경우 무시하도록 처리 (개선 필요)
                print(f"Warning: Skipping single BMS file training for {file_name}. Audio file is required.")
                continue # 다음 파일로 넘어감
                # pairs_to_process = [(None, file_path)] # 오디오 없이 BMS만 처리하는 경우 (별도 모델 필요)

            if not pairs_to_process:
                print(f"Warning: No valid data pairs found for {file_name}. Skipping.")
                # 임시 디렉토리 정리
                if current_temp_dir and os.path.exists(current_temp_dir):
                     shutil.rmtree(current_temp_dir, ignore_errors=True)
                continue # 다음 파일로 넘어감

            # 각 (ogg, bms) 쌍에 대해 학습 데이터 준비 및 학습
            for i, (ogg_path, bms_path) in enumerate(pairs_to_process):
                pair_name = f"{os.path.basename(ogg_path)}_{os.path.basename(bms_path)}" if ogg_path else os.path.basename(bms_path)
                print(f"  - Processing pair {i+1}/{len(pairs_to_process)}: {pair_name}")

                try:
                    # 데이터 준비 (오디오 특징 → 임베딩, BMS → 타겟)
                    features = self.preprocessor.extract_features(ogg_path)
                    if features is None: raise ValueError(f"Could not extract audio features from {ogg_path}")
                    embedding = self.embedding_model.extract_embedding(features)
                    # 임베딩은 (embedding_dim,) 형태

                    targets = parse_bms_to_targets(bms_path)
                    if len(targets) == 0: raise ValueError(f"No valid training targets found in {bms_path}")

                    # 학습 데이터셋 형태 맞추기
                    # X: (샘플 수, 임베딩 차원), y: (샘플 수, 타겟 차원)
                    # 각 타겟 벡터에 대해 동일한 임베딩 사용
                    X_single = np.tile(embedding, (len(targets), 1)) # (len(targets), embedding_dim)
                    y_single = targets # (len(targets), output_dim)

                    print(f"    Prepared {len(y_single)} samples for training pair: {pair_name}")
                    if len(X_single) == 0 or len(y_single) == 0: 
                         print(f"Warning: Empty dataset for pair {pair_name}. Skipping training for this pair.")
                         continue # 데이터가 없으면 학습 건너뛰기

                    # 학습 진행 상황 콜백 업데이트 (파일/쌍 단위)
                    if progress_callback:
                        data = {
                            'progress_type': 'data_loading', # 이제 'data_loading' 상태는 파일 단위
                            'status': f"학습 준비 중: [{file_name}] - {pair_name}",
                            'file_name': file_name, # 현재 처리 중인 파일 이름
                            # 파일 전체 진행률 계산 (처리된 파일 수 / 전체 파일 수)
                            'progress_percent': int((processed_count -1 + (i+1)/len(pairs_to_process)) / total_files_or_pairs * 100), # 파일 내 쌍 진행률 반영
                        }
                        if hasattr(progress_callback, 'emit'):
                            progress_callback.emit(data)
                        elif progress_callback:
                            progress_callback(data)

                    # 모델 학습
                    print(f"    Starting training for pair: {pair_name} for {epochs} epochs.")
                    self.note_prediction_model.model.fit(
                        X_single, y_single,
                        epochs=epochs, # 파일별로 설정된 epochs만큼 학습
                        batch_size=32,
                        validation_split=0.2, # 데이터가 적을 경우 validation_split 조정 또는 제거 필요
                        callbacks=[TrainingProgressCallback(progress_callback, total_epochs=epochs)], # 파일별 epoch 콜백 전달
                        verbose=0 # 학습 상세 로그 비활성화
                    )
                    print(f"    Finished training for pair: {pair_name}.")

                except ValueError as ve:
                    print(f"Error preparing data or training for pair {pair_name}: {ve}")
                    continue # 다음 쌍으로 넘어감
                except Exception as ex:
                    print(f"Unexpected error during training for pair {pair_name}: {ex}")
                    continue # 다음 쌍으로 넘어감

            # 파일 처리 완료 후 임시 디렉토리 정리 (main에서 일괄 처리하도록 변경)
            # if current_temp_dir and os.path.exists(current_temp_dir):
            #      shutil.rmtree(current_temp_dir, ignore_errors=True)

        # 모든 파일 학습 완료 후 최종 모델 저장
        # 모델 저장은 train_model_from_bms_files 함수에서 담당
        print("All files processed. Model training finished.")

    def generate(self, audio_path, bpm, difficulty="normal", model_path=None):
        if model_path and os.path.exists(model_path):
             self.load_model(model_path)

        # 오디오 전처리 및 임베딩
        features = self.preprocessor.extract_features(audio_path)
        embedding = self.embedding_model.extract_embedding(features)

        # 노트 생성 (MusicNoteAI 자체에서 처리)
        notes = [] # 생성된 노트 리스트
        # 임베딩과 BPM/난이도를 활용하여 노트 시퀀스 생성 로직 구현
        # NotePatternGenerator의 generate_notes 로직을 MusicNoteAI로 옮기거나 재구현

        # 간단 예시: MusicNoteAI에서 바로 노트 생성 로직 호출
        # 실제 구현에서는 임베딩과 BPM/난이도를 바탕으로 더 복잡한 패턴 생성
        # 여기서는 NotePatternGenerator의 generate_notes를 임시로 사용
        # NotePatternGenerator는 순수 예측 모델이므로 generate_notes 기능은 MusicNoteAI에 구현해야 함.
        # 현재는 NotePredictionModel로 이름 변경했으므로 generate_notes 기능은 MusicNoteAI에 구현해야 함.
        # 임시로 MusicNoteAI 내에서 노트 생성 로직을 구현합니다.

        beats_per_measure = 4
        seconds_per_beat = 60 / bpm
        audio_duration = librosa.get_duration(path=audio_path) # 오디오 길이 다시 계산 필요
        total_beats = int(audio_duration / seconds_per_beat)

        difficulty_settings = {
            "easy":   {"note_density": 0.15, "hold_chance": 0.07, "boss_chance": 0.03},
            "normal": {"note_density": 0.25, "hold_chance": 0.12, "boss_chance": 0.06},
            "hard":   {"note_density": 0.35, "hold_chance": 0.18, "boss_chance": 0.10}
        }
        settings = difficulty_settings.get(difficulty, difficulty_settings["normal"])
        max_consecutive_notes = 3
        consecutive_notes = 0

        # 노트 생성 로직 개선을 위한 변수 추가
        min_seconds_between_notes = seconds_per_beat * 0.5 # 최소 노트 간격 (예: 8분음표 간격)
        last_note_time = -float('inf') # 마지막 노트가 생성된 시간

        # 임베딩을 사용하여 각 비트 위치의 노트 속성 예측
        # NotePredictionModel은 임베딩->노트 속성 예측 모델이므로,
        # 각 비트 위치마다 임베딩을 입력으로 넣어 예측 결과를 얻음
        # 하지만 MusicEmbeddingModel은 전체 오디오 임베딩을 반환하므로,
        # 각 비트/시간 스텝별 임베딩을 추출하는 방식이 더 적합 (CNN 출력 활용)
        # 현재 MusicEmbeddingModel은 GlobalAveragePooling을 사용하여 시계열 정보 손실
        # 개선 필요: EmbeddingModel이 시계열 임베딩을 반환하도록 수정
        # 여기서는 단순화를 위해 전체 오디오 임베딩을 각 비트에 동일하게 적용

        # 임시 노트 생성 로직 (MusicNoteAI 내에서 예측 모델 활용)
        for t in range(total_beats):
            # 현재 비트의 시간
            current_time = t * seconds_per_beat

            # 임베딩을 예측 모델에 입력하여 노트 속성 예측
            # 예측 모델은 4차원 벡터 [일반, 홀드, 보스, 시각효과] 확률을 반환
            note_prediction = self.note_prediction_model.model.predict(embedding.reshape(1, -1))[0] # (4,) 벡터

            # 디버깅: 예측 결과 확인
            print(f"Time step {t}: Note Prediction Shape: {note_prediction.shape}, Values: {note_prediction}")

            # 예측된 확률을 바탕으로 노트 속성 결정
            # 각 인덱스는 [일반, 홀드, 보스, 시각효과]에 해당
            prob_normal = note_prediction[0] # 일반 노트 확률
            prob_hold = note_prediction[1] # 홀드 노트 확률
            prob_boss = note_prediction[2] # 보스 효과 확률
            prob_visual = note_prediction[3] # 시각 효과 확률

            # 예측된 값과 난이도 설정을 결합하여 최종 노트 결정
            # 간단 예시: 예측된 확률과 랜덤성을 결합하여 노트 생성
            # 난이도 설정을 반영하여 확률 조정 가능

            # 최소 간격 확인
            if current_time - last_note_time < min_seconds_between_notes:
                 consecutive_notes = max(0, consecutive_notes -1) # 간격 부족 시 연속 노트 카운트 감소 (뭉침 방지)
                 continue # 다음 비트로 넘어감

            # 생성할 노트 유형 결정 (우선순위 또는 조합 고려 필요)
            # 여기서는 간단히 예측 확률 기반으로 생성
            note_info = None

            # 홀드 노트 생성 시도
            if prob_hold * np.random.random() * (settings["hold_chance"] * 2) > 0.5 and consecutive_notes < max_consecutive_notes: # 홀드 노트 확률 및 랜덤성, 난이도 가중
                 # 홀드 노트 길이 결정 로직 필요 (현재는 예측 결과에서 얻기 어려움)
                 # 임시로 예측된 확률 기반으로 길이 결정 (개선 필요)
                 # 현재 모델 출력은 홀드 길이 자체를 예측하지 않으므로, 임시로 고정 또는 확률 기반 길이 사용
                 # prob_hold 값을 활용하여 길이 결정 시도 (간단 예시)
                 hold_length_beats = max(1, int(prob_hold * 8)) # 예측 확률에 비례하여 홀드 길이 (최대 8비트 예시)
                 hold_duration = hold_length_beats * seconds_per_beat

                 note_info = {
                     "time": current_time,
                     "is_hold": True,
                     "hold_length": hold_duration, # 길이를 초 단위로 저장 (BMS 변환 시 마디/위치로 계산)
                     "is_boss": False,
                     "visual_effect": prob_visual > 0.5 # 4차원 예측 결과의 마지막 인덱스 사용
                 }
                 last_note_time = current_time + hold_duration * 0.8 # 홀드 노트 끝나는 시간 근처까지 다음 노트 생성 억제 (조정 필요)
                 consecutive_notes += 1

            # 보스 노트 생성 시도 (홀드 노트가 아닐 때)
            elif prob_boss * np.random.random() * (settings["boss_chance"] * 2) > 0.5 and consecutive_notes < max_consecutive_notes:
                 note_info = {
                     "time": current_time,
                     "is_hold": False,
                     "hold_length": 0,
                     "is_boss": True,
                     "visual_effect": prob_visual > 0.5 # 4차원 예측 결과의 마지막 인덱스 사용
                 }
                 last_note_time = current_time # 보스 노트도 간격 적용
                 consecutive_notes += 1

            # 일반 노트 생성 시도 (홀드, 보스 노트가 아닐 때)
            elif prob_normal * np.random.random() * (settings["note_density"] * 2) > 0.5 and consecutive_notes < max_consecutive_notes:
                 note_info = {
                     "time": current_time,
                     "is_hold": False,
                     "hold_length": 0,
                     "is_boss": False,
                     "visual_effect": prob_visual > 0.5 # 4차원 예측 결과의 마지막 인덱스 사용
                 }
                 last_note_time = current_time # 일반 노트도 간격 적용
                 consecutive_notes += 1
            else:
                 consecutive_notes = 0 # 노트 미생성 시 연속 노트 수 초기화

            if note_info:
                 notes.append(note_info)

        return notes

    def save_model(self, path):
        # NotePredictionModel만 저장
        self.note_prediction_model.model.save(path, save_format='keras')

    def load_model(self, path):
        # NotePredictionModel만 로드
        self.note_prediction_model.model = models.load_model(path)

def train_model_from_bms_files(mdm_bms_paths, output_model_path, epochs=10, progress_callback=None):
    """mdm/bms 파일 경로 목록으로부터 모델을 학습하고 저장합니다."""
    try:
        # MusicNoteAI 인스턴스 생성 및 학습 시작
        # temp 디렉토리 처리는 여기서 또는 MusicNoteAI 내에서 관리
        temp_base_dir = tempfile.mkdtemp() # 학습 시작 시 임시 디렉토리 생성

        ai = MusicNoteAI() # AudioPreprocessor, EmbeddingModel 포함
        ai.train(mdm_bms_paths, temp_base_dir, epochs=epochs, progress_callback=progress_callback)
        ai.save_model(output_model_path)

        # 학습 완료 후 임시 디렉토리 정리
        # shutil.rmtree(temp_base_dir, ignore_errors=True)
        # 임시 디렉토리 정리는 main.py의 clear_files와 통합 관리하는 것이 더 좋을 수 있음.
        # 일단 여기서는 생성만 하고 정리는 main에서 하도록 유지.

        return ai

    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise
