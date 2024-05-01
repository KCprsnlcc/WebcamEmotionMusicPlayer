import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QListWidget, QPushButton, QWidget, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

class MusicPlayerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simple Music Player")

        self.playlist = []
        self.current_index = 0

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.listwidget = QListWidget()
        layout.addWidget(self.listwidget)

        load_button = QPushButton("Load Music")
        load_button.clicked.connect(self.load_music)
        layout.addWidget(load_button)

        play_button = QPushButton("Play")
        play_button.clicked.connect(self.play_music)
        layout.addWidget(play_button)

        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self.stop_music)
        layout.addWidget(stop_button)

        central_widget.setLayout(layout)

    def load_music(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Music", "", "MP3 files (*.mp3)")
        for file_path in file_paths:
            self.playlist.append(file_path)
            self.listwidget.addItem(os.path.basename(file_path))

    def play_music(self):
        if self.playlist:
            media_content = QMediaContent()
            media_content.setUrl(QUrl.fromLocalFile(self.playlist[self.current_index]))

            if hasattr(self, "player"):
                self.player.setMedia(media_content)
                self.player.play()
            else:
                self.player = QMediaPlayer()
                self.player.setMedia(media_content)
                self.player.play()

    def stop_music(self):
        if hasattr(self, "player"):
            self.player.stop()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MusicPlayerApp()
    window.show()
    sys.exit(app.exec_())
