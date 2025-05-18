# bms_gui.py

import sys
import numpy as np
if not hasattr(np, 'complex'):
    np.complex = complex
import librosa
import os
import shutil
import zipfile
import tempfile
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog, QVBoxLayout, QMessageBox, QComboBox, QGroupBox, QFormLayout, QSpinBox, QProgressBar, QMainWindow, QTabWidget, QListWidget, QHBoxLayout
)
from PyQt5.QtCore import QThread, pyqtSignal
from learning import MusicNoteAI, train_model_from_bms_files, extract_mdm_pairs
import soundfile as sf

def get_audio_duration(audio_path):
    """음악 파일의 길이를 초 단위로 반환합니다."""
    try:
        f = sf.SoundFile(audio_path)
        duration = len(f) / f.samplerate
        return duration
    except Exception as e:
        print(f"Error getting audio duration for {audio_path}: {str(e)}")
        return 0

def copy_music_file(audio_path, output_dir):
    """음악 파일을 BMS 파일과 같은 디렉토리에 복사하고 파일명을 반환합니다."""
    os.makedirs(output_dir, exist_ok=True)
    _, ext = os.path.splitext(audio_path)
    # music.ogg로 고정 (Muse Dash 규격)
    new_audio_filename = "music.ogg" # BMS에서 참조할 파일명
    new_audio_path = os.path.join(output_dir, new_audio_filename)

    try:
        # librosa를 사용하여 ogg로 변환 후 저장
        y, sr = librosa.load(audio_path, sr=None)
        # soundfile을 사용하여 ogg로 변환 후 저장
        sf.write(new_audio_path, y, sr, format='OGG')
        return new_audio_filename
    except Exception as e:
        print(f"Error converting and copying audio file {audio_path}: {str(e)}")
        # 변환 실패 시 원본 확장자로 시도 (fallback)
        try:
            new_audio_filename = f"music{ext.lower()}"
            new_audio_path = os.path.join(output_dir, new_audio_filename)
            shutil.copy2(audio_path, new_audio_path)
            return new_audio_filename
        except Exception as e2:
            print(f"Fallback copy failed for {audio_path}: {str(e2)}")
            return None # 변환/복사 모두 실패

def write_muse_dash_bms(notes, bpm, output_path, difficulty="normal", metadata=None, music_filename="music.ogg"):
    if metadata is None:
        metadata = {
            "title": "Auto Metal Chart",
            "artist": "AutoGen",
            "genre": "Metal",
            "level_designer": "GPT MuseBot",
            "play_level": 8,
            "rank": 2
        }

    with open(output_path, "w", encoding="utf-8") as f:
        # 헤더 필드
        f.write("*---------------------- HEADER FIELD\n\n")
        f.write("#PLAYER 2\n")
        f.write(f"#GENRE {metadata['genre']}\n")
        f.write(f"#TITLE {metadata['title']}\n")
        f.write(f"#ARTIST {metadata['artist']}\n")
        f.write(f"#LEVELDESIGN {metadata['level_designer']}\n")
        f.write(f"#BPM {int(bpm)}\n")
        f.write(f"#PLAYLEVEL {metadata['play_level']}\n")
        f.write(f"#RANK {metadata['rank']}\n")
        f.write("#LNTYPE 1\n")
        f.write("\n")

        # WAV 슬롯 정의
        # 일반 노트 효과음
        for i in range(1, 10):
            f.write(f"#WAV{format(i, '02X')} Note{i}\n")
        
        # 특수 노트 효과음
        f.write("#WAV0A Large 1\n")
        f.write("#WAV0B Large 2\n")
        f.write("#WAV0C Raider\n")
        f.write("#WAV0D Hammer\n")
        f.write("#WAV0E Gemini\n")
        f.write("#WAV0F Hold\n")
        
        # 배경 음악
        f.write(f"#WAV10 {music_filename}\n")
        
        # 보스 관련 효과음
        f.write("#WAV1A Boss Entrance\n")
        f.write("#WAV1B Boss Exit\n")
        f.write("#WAV1C Boss Ready Phase 1\n")
        f.write("#WAV1D Boss End Phase 1\n")
        f.write("#WAV1E Boss Ready Phase 2\n")
        f.write("#WAV1F Boss End Phase 2\n")
        f.write("#WAV1G Boss Swap Phase 1-2\n")
        f.write("#WAV1H Boss Swap Phase 2-1\n")
        f.write("#WAV1J Hide Notes\n")
        f.write("#WAV1K Unhide Notes\n")
        f.write("#WAV1L Hide Boss\n")
        f.write("#WAV1M Unhide boss\n")
        
        # 시각 효과
        f.write("#WAV2A Scanline Ripples ON\n")
        f.write("#WAV2B Scanline Ripples OFF\n")
        f.write("#WAV2C Chromatic Aberration ON\n")
        f.write("#WAV2D Chromatic Aberration OFF\n")
        f.write("#WAV2E Vignette ON\n")
        f.write("#WAV2F Vignette OFF\n")
        
        f.write("\n*---------------------- MAIN DATA FIELD\n\n")

        # 메인 패턴 데이터
        current_measure = 1
        notes_per_measure = 16
        
        # 채널 정의
        channels = {
            "11": "일반 키 노트",
            "13": "일반 키 노트 (상단)",
            "14": "일반 키 노트 (하단)",
            "15": "홀드 노트",
            "18": "보스 효과",
            "53": "시각 효과 1",
            "54": "시각 효과 2"
        }

        # 보스 패턴 상태
        boss_state = {
            "phase": 1,
            "active": False,
            "duration": 0
        }

        for i, note in enumerate(notes):
            measure = int(i / notes_per_measure) + 1
            position_in_measure = i % notes_per_measure

            # 각 마디의 노트 배열 초기화
            if measure != current_measure:
                current_measure = measure

            # 노트 패턴 생성
            for channel in channels.keys():
                note_line = ["00"] * notes_per_measure
                
                # 채널별 특수 패턴 생성
                if channel == "11":  # 일반 키 노트
                    if position_in_measure % 2 == 0:
                        note_line[position_in_measure] = "01"
                elif channel == "13":  # 상단 노트
                    if position_in_measure % 3 == 0:
                        note_line[position_in_measure] = "02"
                elif channel == "14":  # 하단 노트
                    if position_in_measure % 3 == 1:
                        note_line[position_in_measure] = "03"
                elif channel == "15":  # 홀드 노트
                    if note["is_hold"] and position_in_measure % 4 == 0:
                        # 홀드 노트 길이에 따라 패턴 생성
                        hold_length = min(note["hold_length"], notes_per_measure - position_in_measure)
                        for j in range(hold_length):
                            if position_in_measure + j < notes_per_measure:
                                note_line[position_in_measure + j] = "0F"
                elif channel == "18":  # 보스 효과
                    if note["is_boss"]:
                        if not boss_state["active"]:
                            note_line[position_in_measure] = "1A"  # 보스 등장
                            boss_state["active"] = True
                            boss_state["phase"] = 1
                            boss_state["duration"] = 16
                        elif boss_state["duration"] > 0:
                            if boss_state["duration"] == 8:
                                note_line[position_in_measure] = "1G"  # 페이즈 전환
                                boss_state["phase"] = 2
                            elif boss_state["duration"] == 1:
                                note_line[position_in_measure] = "1B"  # 보스 퇴장
                                boss_state["active"] = False
                            else:
                                note_line[position_in_measure] = "1C" if boss_state["phase"] == 1 else "1E"
                            boss_state["duration"] -= 1
                elif channel == "53":  # 시각 효과 1
                    if note["visual_effect"] and position_in_measure % 16 == 0:
                        note_line[position_in_measure] = "2A"
                elif channel == "54":  # 시각 효과 2
                    if note["visual_effect"] and position_in_measure % 16 == 8:
                        note_line[position_in_measure] = "2B"

            note_string = "".join(note_line)
            f.write(f"#{measure:03d}{channel}:{note_string}\n")

class BMSGeneratorThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, audio_path, output_path, difficulty, metadata, model_path=None):
        super().__init__()
        self.audio_path = audio_path
        self.output_path = output_path
        self.difficulty = difficulty
        self.metadata = metadata
        self.model_path = model_path
        self.ai_generator = MusicNoteAI() # MusicNoteAI 인스턴스 생성

    def run(self):
        try:
            self.progress.emit(10) # 음악 길이 확인 (10%)
            audio_duration = get_audio_duration(self.audio_path)
            if audio_duration == 0: raise ValueError("오디오 파일 길이를 가져올 수 없습니다.")
            self.progress.emit(20)

            # MusicNoteAI를 사용하여 노트 생성
            notes = self.ai_generator.generate(
                self.audio_path, # 오디오 파일 경로
                self.metadata['bpm'],
                self.difficulty,
                model_path=self.model_path # 모델 경로 전달
            )
            self.progress.emit(70) # 노트 생성 (50%)

            output_dir = os.path.dirname(self.output_path)
            music_filename = copy_music_file(self.audio_path, output_dir) # 음악 파일 복사
            if not music_filename: raise ValueError("음악 파일을 복사할 수 없습니다.")
            self.progress.emit(80)

            write_muse_dash_bms(notes, self.metadata['bpm'], self.output_path, self.difficulty, self.metadata, music_filename) # BMS 파일 작성 (20%)
            self.progress.emit(100)

            self.finished.emit({
                "bpm": self.metadata['bpm'],
                "note_count": len(notes),
                "music_file": music_filename,
                "duration": audio_duration
            })
        except Exception as e:
            # 오류 발생 시 스레드에서 오류 시그널을 emit
            print(f"Error in BMSGeneratorThread: {str(e)}") # 터미널에도 오류 출력
            self.error.emit(str(e))

class BMSGeneratorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 메타데이터 입력 섹션
        metadata_group = QGroupBox("메타데이터")
        metadata_layout = QFormLayout()

        self.title_input = QLineEdit()
        self.title_input.setText("Auto Metal Chart")
        metadata_layout.addRow("제목:", self.title_input)

        self.artist_input = QLineEdit()
        self.artist_input.setText("AutoGen")
        metadata_layout.addRow("아티스트:", self.artist_input)

        self.genre_input = QLineEdit()
        self.genre_input.setText("Metal")
        metadata_layout.addRow("장르:", self.genre_input)

        self.level_designer_input = QLineEdit()
        self.level_designer_input.setText("GPT MuseBot")
        metadata_layout.addRow("레벨 디자이너:", self.level_designer_input)

        self.bpm_input = QSpinBox()
        self.bpm_input.setRange(60, 300)
        self.bpm_input.setValue(120)
        metadata_layout.addRow("BPM:", self.bpm_input)

        self.play_level_input = QSpinBox()
        self.play_level_input.setRange(1, 15)
        self.play_level_input.setValue(8)
        metadata_layout.addRow("플레이 레벨:", self.play_level_input)

        self.rank_input = QSpinBox()
        self.rank_input.setRange(1, 3)
        self.rank_input.setValue(2)
        metadata_layout.addRow("랭크:", self.rank_input)

        metadata_group.setLayout(metadata_layout)
        layout.addWidget(metadata_group)

        # 파일 입력 섹션
        file_group = QGroupBox("파일 설정")
        file_layout = QVBoxLayout()

        self.input_label = QLabel("음악 파일 경로 (.ogg):")
        self.input_line = QLineEdit()
        self.input_browse = QPushButton("찾아보기")
        self.input_browse.clicked.connect(self.browse_input)

        self.output_label = QLabel("출력 BMS 파일 경로:")
        self.output_line = QLineEdit()
        self.output_browse = QPushButton("찾아보기")
        self.output_browse.clicked.connect(self.browse_output)

        # 모델 파일 경로 추가
        self.model_label = QLabel("모델 파일 경로 (.keras):")
        self.model_line = QLineEdit()
        self.model_browse = QPushButton("찾아보기")
        self.model_browse.clicked.connect(self.browse_model)

        file_layout.addWidget(self.input_label)
        file_layout.addWidget(self.input_line)
        file_layout.addWidget(self.input_browse)
        file_layout.addWidget(self.output_label)
        file_layout.addWidget(self.output_line)
        file_layout.addWidget(self.output_browse)
        file_layout.addWidget(self.model_label)
        file_layout.addWidget(self.model_line)
        file_layout.addWidget(self.model_browse)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 난이도 설정 섹션
        difficulty_group = QGroupBox("난이도 설정")
        difficulty_layout = QVBoxLayout()

        self.difficulty_label = QLabel("난이도:")
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["easy", "normal", "hard"])

        difficulty_layout.addWidget(self.difficulty_label)
        difficulty_layout.addWidget(self.difficulty_combo)

        difficulty_group.setLayout(difficulty_layout)
        layout.addWidget(difficulty_group)

        # 진행 상태 섹션
        progress_group = QGroupBox("진행 상태")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("대기 중...")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # 실행 버튼
        self.run_button = QPushButton("BMS 생성하기")
        self.run_button.clicked.connect(self.run_generator)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def browse_input(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "음악 파일 선택", "", "오디오 파일 (*.ogg *.mp3 *.wav)")
        if file_name:
            self.input_line.setText(file_name)

    def browse_output(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "BMS 파일 저장", "", "BMS 파일 (*.bms)")
        if file_name:
            self.output_line.setText(file_name)

    def browse_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "모델 파일 선택", "", "Keras 모델 파일 (*.keras)")
        if file_name:
            self.model_line.setText(file_name)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if value < 70:
            self.progress_label.setText("노트 생성 중...")
        elif value < 80:
            self.progress_label.setText("음악 파일 복사 중...")
        elif value < 100:
            self.progress_label.setText("BMS 파일 작성 중...")
        else:
            self.progress_label.setText("완료!")

    def generation_finished(self, result):
        self.run_button.setEnabled(True)
        minutes = int(result['duration'] // 60)
        seconds = int(result['duration'] % 60)
        QMessageBox.information(self, "완료", 
            f"BMS 파일 생성 완료!\n\n"
            f"BPM: {result['bpm']}\n"
            f"생성된 노트 수: {result['note_count']}\n"
            f"음악 파일: {result['music_file']}\n"
            f"음악 길이: {minutes}분 {seconds}초")

    def generation_error(self, error_msg):
        self.run_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("오류 발생")
        QMessageBox.critical(self, "오류 발생", error_msg)

    def run_generator(self):
        audio_path = self.input_line.text()
        output_path = self.output_line.text()
        difficulty = self.difficulty_combo.currentText()
        model_path = self.model_line.text()

        if not audio_path or not output_path:
            QMessageBox.warning(self, "경고", "모든 경로를 입력해 주세요.")
            return

        # 메타데이터 수집
        metadata = {
            "title": self.title_input.text(),
            "artist": self.artist_input.text(),
            "genre": self.genre_input.text(),
            "level_designer": self.level_designer_input.text(),
            "bpm": self.bpm_input.value(),
            "play_level": self.play_level_input.value(),
            "rank": self.rank_input.value()
        }

        # UI 상태 업데이트
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("시작...")

        # 생성 스레드 시작
        self.generator_thread = BMSGeneratorThread(audio_path, output_path, difficulty, metadata, model_path)
        self.generator_thread.progress.connect(self.update_progress)
        self.generator_thread.finished.connect(self.generation_finished)
        self.generator_thread.error.connect(self.generation_error)
        self.generator_thread.start()

class TrainingThread(QThread):
    error = pyqtSignal(str)
    finished = pyqtSignal()
    progress = pyqtSignal(dict)

    def __init__(self, train_func):
        super().__init__()
        self.train_func = train_func

    def run(self):
        try:
            self.train_func(progress_callback=self.progress.emit)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class ModelTrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.temp_dirs = []
        self.training_files = []
        self.training_mdm_files = []

    def init_ui(self):
        layout = QVBoxLayout()

        # 학습 파일 선택 섹션
        file_group = QGroupBox("학습 파일 선택")
        file_layout = QVBoxLayout()

        # 파일 목록 위젯
        self.file_list = QListWidget()
        file_layout.addWidget(self.file_list)

        # 파일 선택 버튼들
        button_layout = QHBoxLayout()
        self.add_files_button = QPushButton("파일 추가")
        self.add_files_button.clicked.connect(self.add_files)
        self.remove_file_button = QPushButton("선택 파일 제거")
        self.remove_file_button.clicked.connect(self.remove_selected_file)
        self.clear_files_button = QPushButton("모든 파일 제거")
        self.clear_files_button.clicked.connect(self.clear_files)

        button_layout.addWidget(self.add_files_button)
        button_layout.addWidget(self.remove_file_button)
        button_layout.addWidget(self.clear_files_button)
        file_layout.addLayout(button_layout)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 학습 설정 섹션
        training_group = QGroupBox("학습 설정")
        training_layout = QFormLayout()

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100)
        self.epochs_input.setValue(10)
        training_layout.addRow("학습 횟수:", self.epochs_input)

        self.model_path_input = QLineEdit()
        self.model_path_input.setText("trained_model.keras")
        self.model_path_browse = QPushButton("저장 위치 선택")
        self.model_path_browse.clicked.connect(self.browse_model_path)
        training_layout.addRow("모델 저장 경로:", self.model_path_input)
        training_layout.addRow("", self.model_path_browse)

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        # 진행 상태 섹션
        progress_group = QGroupBox("진행 상태")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # 학습 상태 표시 레이블들
        self.epoch_label = QLabel("Epoch: -")
        self.loss_label = QLabel("Loss: -")
        self.val_loss_label = QLabel("Validation Loss: -")
        self.time_label = QLabel("경과 시간: -")
        self.progress_label = QLabel("대기 중...")
        self.current_file_label = QLabel("현재 파일: -")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.current_file_label)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.epoch_label)
        progress_layout.addWidget(self.loss_label)
        progress_layout.addWidget(self.val_loss_label)
        progress_layout.addWidget(self.time_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # 학습 버튼
        self.train_button = QPushButton("모델 학습 시작")
        self.train_button.clicked.connect(self.start_training)
        layout.addWidget(self.train_button)

        self.setLayout(layout)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "학습용 BMS/MDM 파일 선택", 
            "", 
            "BMS/MDM 파일 (*.bms *.mdm)"
        )
        if not files:
            return
        for file_path in files:
            if file_path.lower().endswith(".bms"):
                self.training_files.append(file_path)
                self.file_list.addItem(os.path.basename(file_path))
            elif file_path.lower().endswith(".mdm"):
                self.training_mdm_files.append(file_path)
                self.file_list.addItem(os.path.basename(file_path))

    def remove_selected_file(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            row = self.file_list.row(item)
            file_name = item.text()

            # MDM 파일인 경우 목록에서 제거
            mdm_path = next((f for f in self.training_mdm_files if os.path.basename(f) == file_name), None)
            if mdm_path:
                self.training_mdm_files.remove(mdm_path)
            else: # BMS 파일인 경우 목록에서 제거
                bms_path = next((f for f in self.training_files if os.path.basename(f) == file_name), None)
                if bms_path:
                    self.training_files.remove(bms_path)

            self.file_list.takeItem(row)

    def clear_files(self):
        self.file_list.clear()
        self.training_files.clear()
        self.training_mdm_files.clear()
        QMessageBox.information(self, "목록 초기화", "학습 파일 목록이 초기화되었습니다.\n임시 파일은 학습 완료 또는 프로그램 종료 시 정리됩니다.")

    def browse_model_path(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "모델 저장 위치", "", "Keras 모델 파일 (*.keras)")
        if file_name:
            if not file_name.endswith('.keras'):
                file_name += '.keras'
            self.model_path_input.setText(file_name)

    def update_training_progress(self, data):
        """학습 진행 상황을 업데이트합니다."""
        # 데이터 로딩 단계 진행 표시 (MDM 파일 단위)
        if data.get('progress_type') == 'data_loading':
            self.progress_bar.setValue(data.get('progress_percent', 0))
            self.progress_label.setText(data.get('status', '데이터 준비 중...'))
            self.current_file_label.setText(f"현재 파일: {data.get('file_name', '-')}")
            self.epoch_label.setText("Epoch: -") # 학습 정보 초기화
            self.loss_label.setText("Loss: -")
            self.val_loss_label.setText("Validation Loss: -")
            self.time_label.setText("경과 시간: -")

        # 학습 단계 진행 표시 (Epoch, Batch)
        elif data.get('progress_type') in ['batch', 'time', 'epoch_end']:
            # 데이터 로딩 단계 완료 후 프로그레스 바 재설정
            if self.progress_bar.value() < 100: # 데이터 로딩 완료 상태가 아니면
                self.progress_bar.setRange(0, self.epochs_input.value()) # Epoch 수에 맞춰 범위 재설정
                self.progress_bar.setValue(0)
                self.progress_label.setText("모델 학습 중...")
                self.current_file_label.setText("현재 파일: -") # 파일 이름 초기화

            self.epoch_label.setText(f"Epoch: {data.get('epoch', '-')}")
            self.loss_label.setText(f"Loss: {data.get('loss', '-'):.4f}")
            if 'val_loss' in data:
                self.val_loss_label.setText(f"Validation Loss: {data.get('val_loss', '-'):.4f}")

            # 경과 시간 표시
            elapsed_time = data.get('elapsed_time', 0)
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            self.time_label.setText(f"경과 시간: {minutes}분 {seconds}초")

            # 진행 바 업데이트 (Epoch 기준)
            self.progress_bar.setValue(data.get('epoch', 0))

        else: # 기타 상태 메시지
            self.progress_label.setText(data.get('status', '알 수 없는 상태'))
            # self.progress_bar.setValue(0) # 기타 상태 시 프로그레스 바 초기화 여부

    def start_training(self):
        # MDM 파일과 BMS 파일 모두 없으면 경고
        if not self.training_mdm_files and not self.training_files:
            QMessageBox.warning(self, "경고", "학습용 MDM 또는 BMS 파일을 선택해주세요.")
            return

        try:
            self.train_button.setEnabled(False)
            self.progress_label.setText("데이터 준비 중...") # 시작 시 데이터 준비 단계
            self.progress_bar.setRange(0, 100) # 데이터 준비 단계는 0-100
            self.progress_bar.setValue(0)
            self.epoch_label.setText("Epoch: -")
            self.loss_label.setText("Loss: -")
            self.val_loss_label.setText("Validation Loss: -")
            self.time_label.setText("경과 시간: -")
            self.current_file_label.setText("현재 파일: -")

            # 학습 데이터셋 (MDM 파일 경로 목록과 BMS 파일 경로 목록)
            all_files_to_train = self.training_mdm_files + self.training_files

            self.train_thread = TrainingThread(lambda progress_callback:\
                train_model_from_bms_files(\
                    all_files_to_train, # 학습 파일 목록 (MDM + BMS)
                    output_model_path=self.model_path_input.text(), # 모델 저장 경로
                    epochs=self.epochs_input.value(), # epochs는 키워드 인자로 유지\
                    progress_callback=progress_callback # 진행 콜백 전달\
                )\
            )
            self.train_thread.finished.connect(self.training_finished)
            self.train_thread.error.connect(self.training_error)
            # TrainingThread의 progress 시그널을 ModelTrainingTab의 update_training_progress 슬롯에 연결
            self.train_thread.progress.connect(self.update_training_progress)
            self.train_thread.start()

        except Exception as e:
            self.training_error(str(e))

    def training_finished(self):
        self.train_button.setEnabled(True)
        self.progress_label.setText("학습 완료!")
        self.progress_bar.setValue(self.epochs_input.value() if self.epochs_input.value() > 0 else 100) # 학습 완료 시 진행 바 100%
        self.current_file_label.setText("현재 파일: -") # 파일 이름 초기화

        # 학습 완료 후 임시 파일 정리 (main.py에서 일괄 관리)
        for temp_dir in self.temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self.temp_dirs.clear()

        QMessageBox.information(self, "완료", "모델 학습이 완료되었습니다.")

    def training_error(self, error_msg):
        """학습 중 발생한 오류를 처리합니다."""
        self.train_button.setEnabled(True)
        self.progress_label.setText("오류 발생")
        self.current_file_label.setText("현재 파일: -") # 파일 이름 초기화
        QMessageBox.critical(self, "오류", f"학습 중 오류 발생: {error_msg}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("BMS 자동 생성기")
        self.setGeometry(100, 100, 600, 800)

        # 탭 위젯 생성
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # BMS 생성 탭
        self.generator_tab = BMSGeneratorTab()
        self.tabs.addTab(self.generator_tab, "BMS 생성")

        # 모델 학습 탭
        self.training_tab = ModelTrainingTab()
        self.tabs.addTab(self.training_tab, "모델 학습")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
