from style_transfer import Config, StyleTransfer

content_file = "inputs/VCTK-Corpus/wav48/p241/p241_003.wav"
style_file = "inputs/VCTK-Corpus/wav48/p225/p225_002.wav"
config = Config()
config.experiment_name = "first-test"

st = StyleTransfer(config)
st.transfer_style(content_file, style_file)
