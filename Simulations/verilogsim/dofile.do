add wave -position insertpoint  \
sim/:tb_beamforming_helper:A \
sim/:tb_beamforming_helper:initdone \
sim/:tb_beamforming_helper:clock \
sim/:tb_beamforming_helper:Z \

run -all
