import numpy as np
import pyglet

from gym_go import govars, gogame


def draw_circle(x, y, color, radius):
    num_sides = 50
    verts = [x, y]
    colors = list(color)
    for i in range(num_sides + 1):
        verts.append(x + radius * np.cos(i * np.pi * 2 / num_sides))
        verts.append(y + radius * np.sin(i * np.pi * 2 / num_sides))
        colors.extend(color)
    pyglet.graphics.draw(len(verts) // 2, pyglet.gl.GL_TRIANGLE_FAN,
                         ('v2f', verts), ('c3f', colors))


def draw_command_labels(batch, window_width, window_height):
    pyglet.text.Label('Pass (p) | Reset (r) | Exit (e)',
                      font_name='Helvetica',
                      font_size=11,
                      x=20, y=window_height - 20, anchor_y='top', batch=batch, multiline=True, width=window_width)


def draw_info(batch, window_width, window_height, upper_grid_coord, state):
    turn = gogame.turn(state)
    turn_str = 'B' if turn == govars.BLACK else 'W'
    prev_player_passed = gogame.prev_player_passed(state)
    game_ended = gogame.game_ended(state)
    info_label = "Turn: {}\nPassed: {}\nGame: {}".format(turn_str, prev_player_passed,
                                                         "OVER" if game_ended else "ONGOING")

    pyglet.text.Label(info_label, font_name='Helvetica', font_size=11, x=window_width - 20, y=window_height - 20,
                      anchor_x='right', anchor_y='top', color=(0, 0, 0, 192), batch=batch, width=window_width / 2,
                      align='right', multiline=True)

    # Areas
    black_area, white_area = gogame.areas(state)
    pyglet.text.Label("{}B | {}W".format(black_area, white_area), font_name='Helvetica', font_size=16,
                      x=window_width / 2, y=upper_grid_coord + 80, anchor_x='center', color=(0, 0, 0, 192), batch=batch,
                      width=window_width, align='center')


def draw_title(batch, window_width, window_height):
    pyglet.text.Label("Go", font_name='Helvetica', font_size=20, bold=True, x=window_width / 2, y=window_height - 20,
                      anchor_x='center', anchor_y='top', color=(0, 0, 0, 255), batch=batch, width=window_width / 2,
                      align='center')


def draw_grid(batch, delta, board_size, lower_grid_coord, upper_grid_coord):
    label_offset = 20
    left_coord = lower_grid_coord
    right_coord = lower_grid_coord
    ver_list = []
    color_list = []
    num_vert = 0
    for i in range(board_size):
        # horizontal
        ver_list.extend((lower_grid_coord, left_coord,
                         upper_grid_coord, right_coord))
        # vertical
        ver_list.extend((left_coord, lower_grid_coord,
                         right_coord, upper_grid_coord))
        color_list.extend([0.3, 0.3, 0.3] * 4)  # black
        # label on the left
        pyglet.text.Label(str(i),
                          font_name='Courier', font_size=11,
                          x=lower_grid_coord - label_offset, y=left_coord,
                          anchor_x='center', anchor_y='center',
                          color=(0, 0, 0, 255), batch=batch)
        # label on the bottom
        pyglet.text.Label(str(i),
                          font_name='Courier', font_size=11,
                          x=left_coord, y=lower_grid_coord - label_offset,
                          anchor_x='center', anchor_y='center',
                          color=(0, 0, 0, 255), batch=batch)
        left_coord += delta
        right_coord += delta
        num_vert += 4
    batch.add(num_vert, pyglet.gl.GL_LINES, None,
              ('v2f/static', ver_list), ('c3f/static', color_list))


def draw_pieces(batch, lower_grid_coord, delta, piece_r, size, state):
    for i in range(size):
        for j in range(size):
            # black piece
            if state[0, i, j] == 1:
                draw_circle(lower_grid_coord + i * delta, lower_grid_coord + j * delta,
                            [0.05882352963, 0.180392161, 0.2470588237],
                            piece_r)  # 0 for black

            # white piece
            if state[1, i, j] == 1:
                draw_circle(lower_grid_coord + i * delta, lower_grid_coord + j * delta,
                            [0.9754120272] * 3, piece_r)  # 255 for white
