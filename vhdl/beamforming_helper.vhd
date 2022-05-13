-- This is an beamforming_helper VHDL model
-- Initially written by Marko Kosunen
-- Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 16.01.2020 15:51
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;


entity beamforming_helper is
    port( reset : in std_logic;
          A : in  std_logic;
          Z : out std_logic
        );
end beamforming_helper;

architecture rtl of beamforming_helper is
begin
    invert:process(A)
    begin
        Z<=NOT A;
    end process;
end architecture;

