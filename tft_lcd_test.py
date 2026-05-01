import board
import digitalio
from PIL import Image
import adafruit_rgb_display.ili9341 as ili9341

spi = board.SPI()

cs = digitalio.DigitalInOut(board.CE0)
dc = digitalio.DigitalInOut(board.D18)
rst = digitalio.DigitalInOut(board.D23)

disp = ili9341.ILI9341(spi, cs=cs, dc=dc, rst=rst)

image = Image.new("RGB", (disp.width, disp.height), (255, 0, 0))
disp.image(image)