### <img src="/images/banner.png" width=900/>
### Build Instructions 

First install Docker: https://docs.docker.com/engine/install/ubuntu/.

The above link is for a ubuntu base system - if you are using a different distro then install accordingly.
This version of Matador is ran from a GUI. Therefore it requires the X Window System (also known as X11, or simply X). This is a client/server windowing system for bitmap displays. It is implemented on most UNIX-like operating systems and has been ported to many other systems. The X server is the program or dedicated terminal that displays the windows and handles input devices such as keyboards, mice, and touchscreens. The clients are applications.

We are going to use X11 as a shared component between the host system and the docker container. We are going to use the Socket files, which is a UNIX technology that helps the daemon or the services running in the host Linux system to communicate with each other.

