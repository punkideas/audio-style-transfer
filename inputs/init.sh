# Pulls mp3 files and chops them up
# depends on ffmpeg executable 
curl -o obama.mp3 http://www.americanrhetoric.com/mp3clipsXE/barackobama/barackobama2004dncARXE.mp3
mkdir obama
python extract_audio_samples.py obama.mp3 --out_dir obama

curl -o bono.mp3 http://www.americanrhetoric.com/mp3clipsXE/religiousaddresses/bonoprayerbreakfastspeechARXE.mp3
mkdir bono
python extract_audio_samples.py bono.mp3 --out_dir bono

curl -o brewer.mp3 http://www.americanrhetoric.com/mp3clips/politicalspeeches/janbrewersb1070.mp3 
mkdir brewer
python extract_audio_samples.py brewer.mp3 --out_dir brewer

