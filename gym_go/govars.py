ANYONE = None
NOONE = -1

BLACK = 0
WHITE = 1
TURN_CHNL = 2
INVD_CHNL = 3
PASS_CHNL = 4
DONE_CHNL = 5

NUM_CHNLS = 6


class Group:
    def __init__(self):
        self.locations = set()
        self.liberties = set()

    def copy(self):
        groupcopy = Group()
        groupcopy.locations = self.locations.copy()
        groupcopy.liberties = self.liberties.copy()
        return groupcopy

    def __str__(self):
        return f'{self.locations}LOC {self.liberties}LIB'

    def __repr__(self):
        return self.__str__()
