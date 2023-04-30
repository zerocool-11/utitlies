import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc

mycroft_hw = HotwordDetector(
        hotword="pluto",
        reference_file = os.path.join(samples_loc,"/home/pluto/snowboy/op/pluto_ref.json"),
        model= Resnet50_Arc_loss(),
        threshold=0.7,
        relaxation_time=2
    )

mic_stream = SimpleMicStream()
mic_stream.start_stream()

print("Say Pluto ")
while True :
    frame = mic_stream.getFrame()
    result = mycroft_hw.scoreFrame(frame)
    if result==None :
        #no voice activity
        continue
    if(result["match"]):
        print("Wakeword uttered",result["confidence"])
