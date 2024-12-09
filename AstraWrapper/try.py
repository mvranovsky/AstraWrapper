#!/bin/python3

import namelist as nm



cavity = nm.Cavity("cav")
newrun = nm.Newrun("newrun")
output = nm.Output("out")
inp  = nm.Input("inp")
apert = nm.Aperture("ape", 2,2)
quad = nm.Quadrupole("quaddata")

