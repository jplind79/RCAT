"""
    draw_polygon.py

    Draw a simple polygon using matplotlib with mouse event handling

    Copyright (C) 2013 Greg von Winckel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Created: Sat Oct 26 22:04:59 MDT 2013

"""

import matplotlib.pyplot as plt

class Canvas(object):
    def __init__(self,ax):
        self.ax = ax

        # Set limits to unit square
        # self.ax.set_xlim((0,1))
        # self.ax.set_ylim((0,1))

        # turn off axis
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        self.path, = ax.plot([],[],'o-',lw=3)
        self.vert = []
        self.ax.set_title('LEFT: new point, MIDDLE: delete last point, RIGHT: close polygon')

        self.x = []
        self.y = []

        self.mouse_button = {1: self._add_point, 2: self._delete_point, 3: self._close_polygon}

    def set_location(self,event):
        if event.inaxes:
            self.x = event.xdata
            self.y = event.ydata

    def _add_point(self):
        self.vert.append((self.x,self.y))

    def _delete_point(self):
        if len(self.vert)>0:
            self.vert.pop()

    def _close_polygon(self):
        self.vert.append(self.vert[0])

    def update_path(self,event):

        # If the mouse pointer is not on the canvas, ignore buttons
        if not event.inaxes: return

        # Do whichever action correspond to the mouse button clicked
        self.mouse_button[event.button]()

        x = [self.vert[k][0] for k in range(len(self.vert))]
        y = [self.vert[k][1] for k in range(len(self.vert))]
        self.path.set_data(x,y)
        plt.draw()

if __name__ == '__main__':

    fig = plt.figure(1,(8,8))
    ax = fig.add_subplot(111)
    cnv = Canvas(ax)

    plt.connect('button_press_event',cnv.update_path)
    plt.connect('motion_notify_event',cnv.set_location)

    plt.show()
