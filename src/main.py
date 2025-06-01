import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from shadows import (create_no_shadow_grid_mask, 
                     create_full_shadow_grid_mask, 
                     create_cross_shadow_grid_mask,
                     create_top_shadow_grid_mask,
                     create_bottom_shadow_grid_mask)
from shadows import DysonTiles

st.header('Dyson Tile')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Tile Shadow Grid')

    # Create a grid of tiles with different shadow patterns
    shadow_patern = st.selectbox(
        'Shadow Pattern',
        options=['No Shadow', 'Full Shadow', 'Cross Shadow', 'Top Shadow', 'Bottom Shadow'],
        index=0
    )

    # Slider for size of the shade in km
    size_shade_km = st.slider('Shade Size (km)', min_value=500, max_value=5000, value=3000, step=100)

    # Slider for number of tiles
    number_tiles = st.slider('Number of Tiles', min_value=5, max_value=15, value=10, step=1)

    # Slider for opacity of activated tiles
    opacity = st.slider('Opacity of transparent tiles', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Create the shadow grid mask based on the selected pattern
    if shadow_patern == 'No Shadow':
        shadow_grid_mask = create_no_shadow_grid_mask(n_tiles_x=number_tiles, n_tiles_y=number_tiles)
    elif shadow_patern == 'Full Shadow':
        shadow_grid_mask = create_full_shadow_grid_mask(n_tiles_x=number_tiles, n_tiles_y=number_tiles)
    elif shadow_patern == 'Cross Shadow':
        shadow_grid_mask = create_cross_shadow_grid_mask(n_tiles_x=number_tiles, n_tiles_y=number_tiles)
    elif shadow_patern == 'Top Shadow':
        shadow_grid_mask = create_top_shadow_grid_mask(n_tiles_x=number_tiles, n_tiles_y=number_tiles)
    elif shadow_patern == 'Bottom Shadow':
        shadow_grid_mask = create_bottom_shadow_grid_mask(n_tiles_x=number_tiles, n_tiles_y=number_tiles)
    else:
        st.error('Invalid shadow pattern selected.')
        shadow_grid_mask = None

    # Display the shadow grid mask
    if shadow_grid_mask is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(shadow_grid_mask*opacity, cmap='gray', interpolation='nearest')
        ax.set_title('Tile patern visualization')
        ax.set_xlabel('Tile Y Position')
        ax.set_ylabel('Tile Z Position')
        plt.colorbar(ax.imshow(shadow_grid_mask*opacity, cmap='gray', interpolation='nearest'), ax=ax)
        st.pyplot(fig)
        st.write('This is the shadow grid mask visualization showing the transparency of each tile based on the selected shadow pattern and opacity.')
    else:
        st.warning('Please select a shadow pattern to visualize.')
        st.write("No shadow grid available for visualization.")
    


with col2:
    st.subheader('Tile Shadow Visualization')

    dyson_tiles = DysonTiles(shadow_grid_mask, shade_size=size_shade_km*1000, figure_pixel_size=(100, 100))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    dyson_tiles.plot_illumination(ax=ax,opacity_on=0.8)
    st.pyplot(fig)
    
