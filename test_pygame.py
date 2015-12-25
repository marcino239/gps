import pygame as pg
import numpy as np
import sys
import h5py

SCREEN_WIDTH = 227
SCREEN_HEIGHT = 227

LINE_THICKNESS = 3

FPS = 20

# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE  = (  0,   0, 255)
GREEN = (  0, 255,   0)
RED   = (255,   0,   0)

TITLE = 'Pygame demo'

MAX_FRAMES = 10 * FPS
MAX_CHANELS = 3

if __name__ == "__main__":
	
	out_file = None
	if len( sys.argv ) >= 2:
		out_file = sys.argv[1]
	
	pg.init()

	display = pg.display.set_mode( (SCREEN_WIDTH, SCREEN_HEIGHT), pg.DOUBLEBUF )
	pg.display.set_caption( TITLE )
	fpsClock = pg.time.Clock()
	
	font_name = pg.font.get_default_font()
	font = pg.font.SysFont(font_name, 40)

	sc = np.array( [SCREEN_WIDTH, SCREEN_HEIGHT] ) / 2.
	sz = np.array( [SCREEN_WIDTH, SCREEN_HEIGHT] ) * [ 0.4, 0.4 ] 

	t = 0.
	dt = 1. / FPS
	w1 = np.pi
	w2 = 1.5 * np.pi
	
	# video storage
	video_store = np.empty( (MAX_FRAMES, SCREEN_WIDTH, SCREEN_HEIGHT ), dtype=np.int32 )
	
	for i in range( MAX_FRAMES ):
	
		display.fill( BLACK )

		omega1 = w1 * t
		omega2 = w2 * t
	
		xy1 = np.array( [np.sin( omega1 ), np.cos( omega1 ) ] ) * sz + sc
		xy2 = np.array( [np.sin( omega2 ), np.cos( omega2 ) ] ) * sz + sc
		t += dt
		
		pg.draw.line( display, WHITE, xy1.astype( int ), xy2.astype( int ), LINE_THICKNESS )

		# check for quit
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit()

		if out_file is not None:
			video_store[ i, : ] = pg.surfarray.array2d( display )

		#Show FPS
		fps_count = str( int( fpsClock.get_fps() ) )
		fps = font.render( fps_count, True, pg.Color("white") )
		display.blit(fps, (SCREEN_WIDTH - fps.get_size()[0], 0) )

		# display
		pg.display.update()
		fpsClock.tick( FPS )

	if out_file is not None:
		print( 'saving data' )
		f = h5py.File( out_file, 'w' )
		f.create_dataset( 'o', data=video_store )
	else:
		print( 'not saving data' )
