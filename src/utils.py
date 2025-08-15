def extract_audio(video_path, audio_path):
    import moviepy.editor as mp

    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)


def chunk_sentences(transcription, chunk_size=5):
    import nltk
    nltk.download('punkt')
    
    sentences = nltk.sent_tokenize(transcription)
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    
    return chunks