""" 
This module contains function to calculate the shadows cast by a sunshade placed at Lagrange point L1.
The sunshade is modeled as a rectangular array of tiles of adjustable transparency.

The module computes the penumbrera and umbra of the sunshade based on the position of the sun, the shade, and displays the 2D shadow on the earth.
"""

import numpy as np
import matplotlib.pyplot as plt

distance_earth_sun = 1.496e11  # Average distance from Earth to Sun in meters
distance_earth_l1 = 1.5e9  # Distance from Earth to L1 in meters
diameter_sun = 1.3914e9  # Diameter of the Sun in meters
diameter_earth = 1.2756e7  # Diameter of the Earth in meters

position_sun = np.array([0,0,0])  # Position of the Sun at the origin
position_earth = np.array([distance_earth_sun , 0 , 0])  # Position of the Earth at L1
position_l1 = np.array([distance_earth_sun - distance_earth_l1, 0 , 0])  # Position of L1


def create_no_shadow_grid_mask(n_tiles_x=10, n_tiles_y=10):
    """
    Create a grid of tiles with all tiles having no shadow.
    
    Parameters:
    n_tiles_x (int): Number of tiles in the x direction.
    n_tiles_y (int): Number of tiles in the y direction.
    
    Returns:
    np.ndarray: A 2D array representing the shadow grid.
    """
    return np.zeros((n_tiles_y, n_tiles_x), dtype=float)

def create_full_shadow_grid_mask(n_tiles_x=10, n_tiles_y=10):
    """
    Create a grid of tiles with all tiles having full shadow.
    
    Parameters:
    n_tiles_x (int): Number of tiles in the x direction.
    n_tiles_y (int): Number of tiles in the y direction.
    
    Returns:
    np.ndarray: A 2D array representing the shadow grid.
    """
    return np.ones((n_tiles_y, n_tiles_x), dtype=float)

def create_cross_shadow_grid_mask(n_tiles_x=10, n_tiles_y=10):
    """
    Create a grid of tiles with a cross pattern of shadow.
    
    Parameters:
    n_tiles_x (int): Number of tiles in the x direction.
    n_tiles_y (int): Number of tiles in the y direction.
    
    Returns:
    np.ndarray: A 2D array representing the shadow grid.
    """
    mask = np.zeros((n_tiles_y, n_tiles_x), dtype=float)
    mask[n_tiles_y // 2 - max(n_tiles_y//10,1) : n_tiles_y // 2+max(n_tiles_y//10,1), :] = 1.0  # Horizontal line
    mask[:, n_tiles_x // 2 - max(n_tiles_x//10,1) : n_tiles_x //2 + max(n_tiles_x//10,1)] = 1.0  # Vertical line
    return mask






def compute_ray_intersection_with_earth(ray_origin, tile_intercept):
    """
    Compute the intersection of a ray with the Earth.
    The ray is defined by its origin and the point where it intercepts the tile.

    Parameters:
    ray_origin (float): The origin of the ray at the edge of the sun (y or z coordinqte).
    tile_intercept (float): The point where the ray intercepts the tile (y or z coordinqte).

    Returns:
    intersection position (float): The intersection point of the ray with the Earth (y or z coordinqte).
    
    """
    # Calculate the slope of the ray
    slope = (tile_intercept - ray_origin) / (distance_earth_sun - distance_earth_l1)
    
    # Calculate the intersection with the Earth
    intersection_position = ray_origin + slope * distance_earth_sun
    
    return intersection_position


def compute_tile_shadow(tile_position, tile_size, scale_factor=(1, 1)):
    """
    Compute the umbra of a tile based on its position, size, and opacity.
    The umbra is the region where the tile blocks all sunlight.

    Parameters:
    tile_position (tuple): The (y, z) coordinates of the tile's center.
    tile_size (float): The size of the tile (assumed square).

    Returns:
    tuple: The coordinates of the umbra rectangle in the form ((top_y, top_z), (bottom_y, bottom_z)).
    """

    # Calculate max position of the tile in the y direction
    tile_top_y = tile_position[0] + tile_size / 2

    # Calculate min position of the tile in the y direction
    tile_bottom_y = tile_position[0] - tile_size / 2

    # Calculate max position of the tile in the z direction
    tile_top_z = tile_position[1] + tile_size / 2

    # Calculate min position of the tile in the z direction
    tile_bottom_z = tile_position[1] - tile_size / 2

    # Calculate the intersection of the ray with the Earth
    umbra_top_y = max(
        compute_ray_intersection_with_earth(diameter_sun / 2, tile_top_y),
        compute_ray_intersection_with_earth(-diameter_sun / 2, tile_bottom_y),
        )
    
    umbra_bottom_y = min(
        compute_ray_intersection_with_earth(diameter_sun / 2, tile_top_y),
        compute_ray_intersection_with_earth(-diameter_sun / 2, tile_bottom_y)
        )
    
    umbra_top_z = max(  
        compute_ray_intersection_with_earth(diameter_sun / 2, tile_top_z),
        compute_ray_intersection_with_earth(-diameter_sun / 2, tile_bottom_z),
        )
    
    umbra_bottom_z = min(
        compute_ray_intersection_with_earth(diameter_sun / 2, tile_top_z),
        compute_ray_intersection_with_earth(-diameter_sun / 2, tile_bottom_z)
        )
    
    penumbra_top_y = compute_ray_intersection_with_earth(diameter_sun / 2, tile_bottom_y)
    penumbra_bottom_y = compute_ray_intersection_with_earth(-diameter_sun / 2, tile_top_y)
    penumbra_top_z = compute_ray_intersection_with_earth(diameter_sun / 2, tile_bottom_z)
    penumbra_bottom_z = compute_ray_intersection_with_earth(-diameter_sun / 2, tile_top_z)

    # Scale the coordinates based on the scale factor
    umbra_top_y *= scale_factor[0]
    umbra_bottom_y *= scale_factor[0]
    umbra_top_z *= scale_factor[1]
    umbra_bottom_z *= scale_factor[1]

    penumbra_top_y *= scale_factor[0]
    penumbra_bottom_y *= scale_factor[0]
    penumbra_top_z *= scale_factor[1]
    penumbra_bottom_z *= scale_factor[1]

    # Return the umbra and penumbra coordinates
    return {
        "umbra": ((umbra_top_y, umbra_top_z), (umbra_bottom_y, umbra_bottom_z)),
        "penumbra": ((penumbra_top_y, penumbra_top_z), (penumbra_bottom_y, penumbra_bottom_z))
    }

class DysonTiles:
    """
    Class to represent a grid of Dyson tiles and compute their shadows.
    Each tile can be configured with its position, size, and opacity.
    """

    def __init__(self, mask, shade_size=1e3, figure_pixel_size=(100, 100)):
        self.n_tiles_y = mask.shape[0]
        self.n_tiles_z = mask.shape[1]
        self.tile_size = shade_size // self.n_tiles_y  # Size of each tile in meters
        self.mask = mask
        self.scale_factor = (figure_pixel_size[0] / diameter_earth, figure_pixel_size[1] / diameter_earth)

        
    def plot_shadow(self, ax=None, opacity_on=0.8):
        """
        Plot the shadow of the tiles on a 2D grid.
        
        Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on. If None, a new figure and axes are created.
        
        Returns:
        matplotlib.axes.Axes: The axes with the plotted shadow.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        Umbras = []
        Penumbras = []
        for y in range(self.n_tiles_y):
            for z in range(self.n_tiles_z):
                if self.mask[y, z] > 0:
                    tile_position = ((self.n_tiles_y//2 - y + 0.5)*self.tile_size, (self.n_tiles_z//2 - z + 0.5)*self.tile_size)
                    shadow = compute_tile_shadow(tile_position, self.tile_size, self.scale_factor)
                    umbra = shadow["umbra"]
                    penumbra = shadow["penumbra"]

                    # Plot the penumbra
                    ax.fill_betweenx([penumbra[0][1], penumbra[1][1]], penumbra[0][0], penumbra[1][0], color='gray', alpha=opacity_on * 0.5, label='Penumbra' if y == 0 and z == 0 else "")
            

                    Umbras.append(umbra)        
                    Penumbras.append(penumbra)

                else:
                    Umbras.append(-1)
                    Penumbras.append(-1)

        for y in range(self.n_tiles_y):
                for z in range(self.n_tiles_z):
                    umbra = Umbras[y * self.n_tiles_z + z]

                    if umbra != -1 and penumbra != -1:
                        # Plot the umbra
                        ax.fill_betweenx([umbra[0][1], umbra[1][1]], umbra[0][0], umbra[1][0], color='black', alpha=opacity_on, label='Umbra' if y == 0 and z == 0 else "")
        # Plot the Earth
        earth_circle = plt.Circle((0, 0), diameter_earth * self.scale_factor[0] / 2, color='blue', alpha=0.3, label='Earth')
        ax.add_artist(earth_circle)            
        ax.set_xlim(-diameter_earth*self.scale_factor[0] / 2 - 30, diameter_earth*self.scale_factor[0] / 2 + 30)
        ax.set_ylim(-diameter_earth*self.scale_factor[1] / 2 - 30, diameter_earth*self.scale_factor[1] / 2 + 30)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Dyson Tile Shadow')
        ax.set_xlabel('Y Coordinate (m)')
        ax.set_ylabel('Z Coordinate (m)')
        ax.grid(True)
        ax.legend()
        return ax

    

if __name__ == "__main__":
    # Example usage
    n_tiles_x = 10
    n_tiles_y = 10
    mask = create_cross_shadow_grid_mask(n_tiles_x, n_tiles_y)
    
    dyson_tiles = DysonTiles(mask, shade_size=3000e3, figure_pixel_size=(100, 100))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    dyson_tiles.plot_shadow(ax=ax,opacity_on=0.8)
    plt.show()
    # Uncomment the following lines to save the figure
    # fig.savefig('dyson_tile_shadow.png', dpi=300, bbox_inches='tight')
