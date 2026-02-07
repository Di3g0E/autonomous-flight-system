"""
Collision Detection System for Quadrotor Environment

This module provides collision detection capabilities using Panda3D's collision system.
It can be optionally integrated with the quadrotor environment without creating dependencies.
"""

import numpy as np

# Make Panda3D imports optional
try:
    from panda3d.core import CollisionSphere, CollisionNode, CollisionHandlerQueue
    from panda3d.core import CollisionTraverser, BitMask32
    PANDA3D_AVAILABLE = True
except ImportError:
    PANDA3D_AVAILABLE = False
    print("Warning: Panda3D not available. Collision detection will be disabled.")


class CollisionDetector:
    """
    Collision detection system for the quadrotor.
    
    This class manages collision detection between the quadrotor and obstacles
    in the environment using Panda3D's collision system.
    """
    
    def __init__(self, render, quad_model, collision_radius=0.3):
        """
        Initialize the collision detector.
        
        Args:
            render: Panda3D render node
            quad_model: The quadrotor model node
            collision_radius: Radius of the collision sphere around the quadrotor (meters)
        """
        self.render = render
        self.quad_model = quad_model
        self.collision_radius = collision_radius
        
        # Collision state
        self.has_collision = False
        self.collision_point = None
        self.collision_normal = None
        self.collision_object_name = None
        
        # Setup collision system
        self._setup_collision_system()
    
    def _setup_collision_system(self):
        """Setup the Panda3D collision detection system."""
        
        # Create collision traverser
        self.collision_traverser = CollisionTraverser('collision_traverser')
        
        # Create collision handler
        self.collision_handler = CollisionHandlerQueue()
        
        # Create collision sphere for the quadrotor
        self.collision_sphere = CollisionSphere(0, 0, 0, self.collision_radius)
        
        # Create collision node
        self.collision_node = CollisionNode('quadrotor_collision')
        self.collision_node.addSolid(self.collision_sphere)
        
        # Set collision masks
        # The quadrotor collides with objects in the 'obstacle' group
        self.collision_node.setFromCollideMask(BitMask32.bit(1))
        self.collision_node.setIntoCollideMask(BitMask32.allOff())
        
        # Attach collision node to quadrotor model
        self.collision_node_path = self.quad_model.attachNewNode(self.collision_node)
        
        # Add to traverser
        self.collision_traverser.addCollider(self.collision_node_path, self.collision_handler)
        
        # For debugging: show collision sphere
        # self.collision_node_path.show()
    
    def check_collisions(self):
        """
        Check for collisions and update collision state.
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        # Clear previous collision state
        self.has_collision = False
        self.collision_point = None
        self.collision_normal = None
        self.collision_object_name = None
        
        # Traverse collision system
        self.collision_traverser.traverse(self.render)
        
        # Check if any collisions occurred
        if self.collision_handler.getNumEntries() > 0:
            self.has_collision = True
            
            # Sort entries by distance (closest first)
            self.collision_handler.sortEntries()
            
            # Get the first (closest) collision
            entry = self.collision_handler.getEntry(0)
            
            # Store collision information
            self.collision_point = entry.getSurfacePoint(self.render)
            self.collision_normal = entry.getSurfaceNormal(self.render)
            self.collision_object_name = entry.getIntoNodePath().getName()
        
        return self.has_collision
    
    def get_collision_info(self):
        """
        Get detailed information about the current collision.
        
        Returns:
            dict: Dictionary containing collision information
        """
        if not self.has_collision:
            return {
                'has_collision': False,
                'collision_point': None,
                'collision_normal': None,
                'collision_object': None,
                'distance_to_collision': None
            }
        
        # Convert Panda3D vectors to numpy arrays
        collision_point = np.array([
            self.collision_point.getX(),
            self.collision_point.getY(),
            self.collision_point.getZ()
        ])
        
        collision_normal = np.array([
            self.collision_normal.getX(),
            self.collision_normal.getY(),
            self.collision_normal.getZ()
        ])
        
        # Get quadrotor position
        quad_pos = self.quad_model.getPos()
        quad_pos_array = np.array([quad_pos.getX(), quad_pos.getY(), quad_pos.getZ()])
        
        # Calculate distance to collision point
        distance = np.linalg.norm(collision_point - quad_pos_array)
        
        return {
            'has_collision': True,
            'collision_point': collision_point,
            'collision_normal': collision_normal,
            'collision_object': self.collision_object_name,
            'distance_to_collision': distance
        }
    
    def reset(self):
        """Reset collision state."""
        self.has_collision = False
        self.collision_point = None
        self.collision_normal = None
        self.collision_object_name = None
    
    def set_collision_radius(self, radius):
        """
        Update the collision sphere radius.
        
        Args:
            radius: New radius in meters
        """
        self.collision_radius = radius
        self.collision_sphere.setRadius(radius)
    
    def enable_debug_visualization(self):
        """Show the collision sphere for debugging."""
        self.collision_node_path.show()
    
    def disable_debug_visualization(self):
        """Hide the collision sphere."""
        self.collision_node_path.hide()


class ObstacleManager:
    """
    Manager for creating and managing collision obstacles in the environment.
    """
    
    def __init__(self, render):
        """
        Initialize the obstacle manager.
        
        Args:
            render: Panda3D render node
        """
        self.render = render
        self.obstacles = []
    
    def add_box_obstacle(self, position, size, name="box_obstacle"):
        """
        Add a box-shaped obstacle to the environment.
        
        Args:
            position: Tuple (x, y, z) for obstacle position
            size: Tuple (width, depth, height) for obstacle dimensions
            name: Name identifier for the obstacle
        
        Returns:
            NodePath: The created obstacle node
        """
        from panda3d.core import CollisionBox, Point3
        
        # Create collision box
        min_point = Point3(-size[0]/2, -size[1]/2, -size[2]/2)
        max_point = Point3(size[0]/2, size[1]/2, size[2]/2)
        collision_box = CollisionBox(min_point, max_point)
        
        # Create collision node
        collision_node = CollisionNode(name)
        collision_node.addSolid(collision_box)
        
        # Set collision masks (obstacles are in group 1)
        collision_node.setFromCollideMask(BitMask32.allOff())
        collision_node.setIntoCollideMask(BitMask32.bit(1))
        
        # Attach to render and set position
        obstacle_np = self.render.attachNewNode(collision_node)
        obstacle_np.setPos(*position)
        
        self.obstacles.append(obstacle_np)
        
        return obstacle_np
    
    def add_sphere_obstacle(self, position, radius, name="sphere_obstacle"):
        """
        Add a sphere-shaped obstacle to the environment.
        
        Args:
            position: Tuple (x, y, z) for obstacle position
            radius: Radius of the sphere
            name: Name identifier for the obstacle
        
        Returns:
            NodePath: The created obstacle node
        """
        # Create collision sphere
        collision_sphere = CollisionSphere(0, 0, 0, radius)
        
        # Create collision node
        collision_node = CollisionNode(name)
        collision_node.addSolid(collision_sphere)
        
        # Set collision masks
        collision_node.setFromCollideMask(BitMask32.allOff())
        collision_node.setIntoCollideMask(BitMask32.bit(1))
        
        # Attach to render and set position
        obstacle_np = self.render.attachNewNode(collision_node)
        obstacle_np.setPos(*position)
        
        self.obstacles.append(obstacle_np)
        
        return obstacle_np
    
    def add_model_collision(self, model_node, name="model_obstacle"):
        """
        Add collision detection to an existing 3D model.
        
        Args:
            model_node: The model NodePath
            name: Name identifier for the obstacle
        
        Returns:
            NodePath: The collision node
        """
        from panda3d.core import CollisionPolygon, GeomNode
        
        # This is a simplified version - for complex models,
        # you might want to use automatic collision mesh generation
        # or manually defined collision shapes
        
        # For now, we'll create a bounding box around the model
        bounds = model_node.getTightBounds()
        if bounds:
            min_point, max_point = bounds
            center = (min_point + max_point) / 2
            size = max_point - min_point
            
            # Create collision box
            from panda3d.core import CollisionBox
            collision_box = CollisionBox(min_point, max_point)
            
            collision_node = CollisionNode(name)
            collision_node.addSolid(collision_box)
            
            # Set collision masks
            collision_node.setFromCollideMask(BitMask32.allOff())
            collision_node.setIntoCollideMask(BitMask32.bit(1))
            
            # Attach to model
            obstacle_np = model_node.attachNewNode(collision_node)
            
            self.obstacles.append(obstacle_np)
            
            return obstacle_np
        
        return None
    
    def clear_obstacles(self):
        """Remove all obstacles from the environment."""
        for obstacle in self.obstacles:
            obstacle.removeNode()
        self.obstacles.clear()
    
    def get_obstacle_count(self):
        """Get the number of obstacles in the environment."""
        return len(self.obstacles)
    
    def enable_debug_visualization(self):
        """Show all obstacle collision shapes for debugging."""
        for obstacle in self.obstacles:
            obstacle.show()
    
    def disable_debug_visualization(self):
        """Hide all obstacle collision shapes."""
        for obstacle in self.obstacles:
            obstacle.hide()
