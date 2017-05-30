from style_transfer import Config, StyleTransfer

content_file = "inputs/test/content.wav"
style_file = "/inputs/test/style.wav"
config = Config()
config.experiment_name = "first-test"

st = StyleTransfer(config)
st.transfer_style(content_file, style_file)
