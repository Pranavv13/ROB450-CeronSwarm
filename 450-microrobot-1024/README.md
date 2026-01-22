
# microrobot-1024

**Authors:** Dong Heon Han, SAM Lab (University of Michigan Robotics)

**The first project on a 1024-magnet array!**  
A scalable 1024-channel electromagnetic array system for programmable collective robotics.

![image 38](https://github.com/user-attachments/assets/deb2ee7f-754a-44dd-9269-846e5b871755)

---

## Overview

This project implements a 1024-magnet array for high-resolution magnetic field control.  
The system is designed for scalability, modularity, and real-time operation.

---

## Repository Structure

```text
firmware/       # Embedded code running on the microcontrollers (Raspberry Pi Pico 1 / Pico 2)
hardware/       # CAD models, PCB layouts, and schematics
software/       # Host-side communication protocols between PC and Pico
documents/      # Design notes, protocols, and system-level documentation
models/         # Analytical magnetic field and system models
simulations/    # Offline numerical simulations based on the models
