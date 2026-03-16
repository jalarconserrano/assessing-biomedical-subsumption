#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:06:15 2024

@author: david
"""
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from lib.ag_relations import EvaluatorMy

def getsimplifiedmodelname(modelname): 
    if modelname.startswith('ollama'):
        return modelname.split(':')[0].split('/')[1]
    return modelname

def confusionmatrix_chart(modelname,confusionmatrix):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusionmatrix,
        display_labels=EvaluatorMy.classlabels
    )
    disp.plot()
    
    plt.ylabel('True label',fontsize=13)
    plt.xlabel('Predicted label', fontsize=13)
    plt.suptitle(f'Model: {modelname}',fontsize=18)

def radar_chart(namemodels,dataframe,labels,title="Radar Chart"):
    # Number of variables we're plotting (based on 'Metric' column)    
    num_vars = len(labels)
    
    colors=[
        '#60A917','#D80073','#1BA1E2','#FA6800',
        '#647687','#E3C800','#A0522D','#76608A'
        ]
    
    angles = radar_factory(num_vars,frame='polygon')
    
    # Set up the figure and polar subplot
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))

    ax.set_rgrids([.25, .50, .75])
    
    for model,color in zip(namemodels,colors):
        ax.plot(angles, dataframe[model], linewidth=2, alpha=0.5, label=model, color=color)
        ax.fill(angles, dataframe[model], facecolor=color, alpha=0.25, label='_nolegend_')
  
    # Add labels for each metric on the outer edge
    #ax.set_xticks(angles)
    #ax.set_xticklabels(labels)
    
    ax.set_varlabels(labels)
    legend = ax.legend(labels, loc=(0.80,0.85),
                              labelspacing=0.1, fontsize='small')
    # Set title and legend
    ax.set_title(title, size=15, color='black', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

def radar_factory(num_vars, frame='circle'):
    """
    get: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
        frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.
    """

    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta



