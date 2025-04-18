# Import necessary libraries
from manim import *
import numpy as np
from typing import List, Tuple, Dict, Optional  # For type hinting

# Set numpy print options for cleaner matrix/vector output in the terminal (doesn't affect Manim output)
np.set_printoptions(precision=3, suppress=True)


class SymmetricMatrixTransformations(Scene):
    """
    Manim scene illustrating the properties of 2D symmetric matrix transformations.
    Demonstrates how symmetric matrices stretch/squish space along orthogonal
    eigenvector directions, preserving their perpendicularity.
    """

    def construct(self) -> None:
        """
        Constructs the animations for the scene.
        """
        # --- Configuration ---
        intro_wait_time: float = 2.0  # Wait time after initial setup
        transform_wait_time: float = 3.5  # Wait time after a transformation animation
        step_wait_time: float = (
            2.0  # Wait time between displaying info and transforming
        )
        matrix_scale: float = 0.7  # Scale factor for matrix display
        text_scale: float = 0.6  # Scale factor for descriptive text
        eigen_scale: float = 0.5  # Scale factor for eigenvalue/vector text
        vector_color_1: ManimColor = YELLOW  # Color for the first eigenvector
        vector_color_2: ManimColor = RED  # Color for the second eigenvector

        # --- Helper Function: Format Math Objects ---
        def format_math(
            obj: np.ndarray | List[float] | float | int, scale: float = 1.0
        ) -> MathTex:
            """
            Formats numpy arrays (as column vectors) or numbers into MathTex objects.

            Args:
                obj: The numpy array, list, float, or int to format.
                scale: The scaling factor for the MathTex font size.

            Returns:
                A MathTex object representing the input.
            """
            font_size = DEFAULT_FONT_SIZE * scale
            if isinstance(obj, (np.ndarray, list)):
                # Ensure it's a column vector for display purposes
                if isinstance(obj, list):
                    obj = np.array(obj)
                obj = obj.reshape(-1, 1)  # Reshape to ensure it's a column
                elements = [f"{x:.2f}" for x in np.ravel(obj)]
                tex_str = (
                    r"\begin{bmatrix} " + r" \\ ".join(elements) + r" \end{bmatrix}"
                )
                return MathTex(tex_str, font_size=font_size)
            elif isinstance(obj, (int, float, np.number)):
                return MathTex(f"{obj:.2f}", font_size=font_size)
            else:
                # Fallback for unexpected types
                return MathTex(str(obj), font_size=font_size)

        # --- Helper Function: Show Transformation ---
        def show_transformation(
            title_text: str,
            matrix_np: np.ndarray,
            plane: NumberPlane,
            show_eigenvectors: bool = True,
        ) -> NumberPlane:
            """
            Displays transformation information (matrix, title, eigenstuff),
            visualizes eigenvectors (optional), and animates the plane transformation.

            Args:
                title_text: Text describing the transformation type.
                matrix_np: The 2x2 numpy array representing the transformation matrix.
                plane: The current NumberPlane mobject.
                show_eigenvectors: Whether to calculate, display, and animate eigenvectors.

            Returns:
                The transformed NumberPlane mobject.
            """
            # --- Display Matrix and Title ---
            title = (
                Text(title_text, font_size=DEFAULT_FONT_SIZE * text_scale)
                .to_corner(UL)
                .shift(DOWN * 0.5)
            )
            matrix_tex = (
                Matrix(matrix_np, h_buff=1.2)
                .scale(matrix_scale)
                .next_to(title, DOWN, buff=0.3)
                .align_to(title, LEFT)
            )
            matrix_label = (
                Text("Matrix:", font_size=DEFAULT_FONT_SIZE * eigen_scale)
                .next_to(matrix_tex, UP, buff=0.1)
                .align_to(matrix_tex, LEFT)
            )

            self.play(Write(title))
            self.play(Write(matrix_label), Create(matrix_tex))
            self.wait(step_wait_time * 0.5)

            # --- Calculate and Display Eigenvalues/Eigenvectors ---
            eigen_label = (
                Text(
                    "Eigenvalues / Eigenvectors:",
                    font_size=DEFAULT_FONT_SIZE * eigen_scale,
                )
                .next_to(matrix_tex, DOWN, buff=0.4)
                .align_to(matrix_tex, LEFT)
            )
            eigen_anims: List[Animation] = [
                Write(eigen_label)
            ]  # List to hold animations for eigenstuff text
            eigen_mobjects = VGroup(
                title, matrix_label, matrix_tex, eigen_label
            )  # Group text for easy fading

            vector_anims: List[Animation] = (
                []
            )  # List to hold animations for creating vectors
            original_eigenvectors_group: Optional[VGroup] = (
                None  # Group to hold visual vectors and labels
            )
            eigenvalues: Optional[np.ndarray] = None
            eigenvectors_matrix: Optional[np.ndarray] = None
            ev1_vec: Optional[np.ndarray] = None
            ev2_vec: Optional[np.ndarray] = None

            if show_eigenvectors:
                try:
                    # Calculate eigenvalues and eigenvectors
                    eigenvalues, eigenvectors_matrix = np.linalg.eig(matrix_np)

                    # --- Sort eigenvalues and corresponding eigenvectors ---
                    # This makes visualization consistent (e.g., largest eigenvalue first)
                    idx = eigenvalues.argsort()[
                        ::-1
                    ]  # Get indices for sorting descending
                    eigenvalues = eigenvalues[idx]
                    eigenvectors_matrix = eigenvectors_matrix[
                        :, idx
                    ]  # Rearrange columns (eigenvectors)

                    # --- Check and Display Orthogonality (Characteristic of Symmetric Matrices) ---
                    ev1_vec = eigenvectors_matrix[:, 0]
                    ev2_vec = eigenvectors_matrix[:, 1]
                    dot_prod = np.dot(ev1_vec, ev2_vec)
                    # Use a small tolerance for floating point comparison
                    ortho_color = (
                        GREEN if np.abs(dot_prod) < 1e-8 else YELLOW
                    )  # Yellow if not perfectly zero
                    ortho_text = (
                        Text(
                            f"Eigenvector dot product: {dot_prod:.2f}",
                            font_size=DEFAULT_FONT_SIZE * eigen_scale * 0.8,
                            color=ortho_color,
                        )
                        .next_to(eigen_label, DOWN, buff=0.8)
                        .align_to(eigen_label, LEFT)
                    )
                    ortho_label = (
                        Text(
                            "(Should be 0 for symmetric matrices)",
                            font_size=DEFAULT_FONT_SIZE * eigen_scale * 0.6,
                        )
                        .next_to(ortho_text, DOWN, buff=0.1)
                        .align_to(ortho_text, LEFT)
                    )

                    eigen_anims.append(Write(ortho_text))
                    eigen_anims.append(Write(ortho_label))
                    eigen_mobjects.add(ortho_text, ortho_label)

                    # --- Prepare Eigenvalue/Eigenvector Text Display ---
                    ev1_val_tex = (
                        format_math(eigenvalues[0], eigen_scale)
                        .next_to(eigen_label, DOWN, buff=0.2)
                        .align_to(eigen_label, LEFT)
                        .shift(RIGHT * 0.5)
                    )
                    ev1_vec_tex = format_math(ev1_vec, eigen_scale).next_to(
                        ev1_val_tex, RIGHT, buff=0.5
                    )

                    ev2_val_tex = (
                        format_math(eigenvalues[1], eigen_scale)
                        .next_to(ev1_val_tex, DOWN, buff=0.3)
                        .align_to(ev1_val_tex, LEFT)
                    )
                    ev2_vec_tex = format_math(ev2_vec, eigen_scale).next_to(
                        ev2_val_tex, RIGHT, buff=0.5
                    )

                    eigenvalue_texts = [ev1_val_tex, ev2_val_tex]
                    eigenvector_texts = [ev1_vec_tex, ev2_vec_tex]
                    eigen_anims.extend(
                        [Write(t) for t in eigenvalue_texts + eigenvector_texts]
                    )
                    eigen_mobjects.add(*eigenvalue_texts, *eigenvector_texts)

                    # --- Create Visual Eigenvectors on the Plane ---
                    # Use plane.c2p (coords_to_point) to convert numpy coords to Manim screen coords.
                    # Vectors start at the origin plane.c2p(0,0) == plane.get_origin()
                    # We need to unpack the vector components for c2p: *ev1_vec -> ev1_vec[0], ev1_vec[1]
                    v1_end_point = plane.c2p(*ev1_vec)
                    v2_end_point = plane.c2p(*ev2_vec)
                    v1 = Vector(
                        v1_end_point - plane.get_origin(), color=vector_color_1
                    ).shift(plane.get_origin())
                    v2 = Vector(
                        v2_end_point - plane.get_origin(), color=vector_color_2
                    ).shift(plane.get_origin())

                    # Create labels that will move with the vectors
                    v1_label = MathTex(
                        r"\vec{v}_1",
                        font_size=DEFAULT_FONT_SIZE * eigen_scale,
                        color=vector_color_1,
                    )
                    v1_label.add_updater(
                        lambda m: m.next_to(v1.get_end(), UR, buff=0.1)
                    )  # Label follows vector tip

                    v2_label = MathTex(
                        r"\vec{v}_2",
                        font_size=DEFAULT_FONT_SIZE * eigen_scale,
                        color=vector_color_2,
                    )
                    v2_label.add_updater(
                        lambda m: m.next_to(v2.get_end(), UR, buff=0.1)
                    )  # Label follows vector tip

                    original_eigenvectors_group = VGroup(
                        v1, v2, v1_label, v2_label
                    )  # Group vectors and labels
                    vector_anims = [
                        Create(v1),
                        Create(v2),
                        Write(v1_label),
                        Write(v2_label),
                    ]

                except np.linalg.LinAlgError:
                    error_text = Text(
                        "Could not compute eigenvalues/eigenvectors.",
                        color=RED,
                        font_size=DEFAULT_FONT_SIZE * eigen_scale,
                    ).next_to(eigen_label, DOWN, buff=0.2)
                    eigen_anims.append(Write(error_text))
                    eigen_mobjects.add(error_text)
                    show_eigenvectors = (
                        False  # Cannot proceed with vector visualization
                    )

            # --- Animate Text Appearance ---
            self.play(LaggedStart(*eigen_anims, lag_ratio=0.2))
            # --- Animate Vector Appearance ---
            if vector_anims:  # Only play if vectors were created
                self.play(LaggedStart(*vector_anims, lag_ratio=0.2))
            self.wait(step_wait_time)

            # --- Apply Transformation Animation ---
            # ApplyMatrix transforms the coordinate system (plane)
            # It also transforms mobjects *within* that coordinate system if they are passed as arguments
            transform_anims: List[Animation] = [ApplyMatrix(matrix_np, plane)]
            if show_eigenvectors and original_eigenvectors_group:
                # Animate the eigenvectors being scaled by the transformation
                # The vectors stretch/squish along their original directions
                transform_anims.append(
                    ApplyMatrix(matrix_np, original_eigenvectors_group)
                )

            self.play(*transform_anims, run_time=3)
            # Vectors automatically update their labels due to the updaters attached earlier
            self.wait(transform_wait_time)

            # --- Cleanup ---
            # Remove text elements
            cleanup_anims: List[Animation] = [FadeOut(eigen_mobjects)]
            # Remove vector elements
            if original_eigenvectors_group:
                # Need to clear updaters before fading otherwise errors can occur
                v1_label.clear_updaters()
                v2_label.clear_updaters()
                cleanup_anims.append(FadeOut(original_eigenvectors_group))

            self.play(*cleanup_anims)

            return plane  # Return the (now transformed) plane object

        # ==================================================================
        #                     SCENE CONSTRUCTION STARTS HERE
        # ==================================================================

        # --- Scene Setup: Initial Grid ---
        plane = NumberPlane(
            x_range=[-7, 7, 1],  # X-axis range [min, max, step]
            y_range=[-5, 5, 1],  # Y-axis range [min, max, step]
            x_length=14,  # Width on screen
            y_length=10,  # Height on screen
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 2,
                "stroke_opacity": 0.6,
            },
            axis_config={
                "include_numbers": True,  # Show numbers on axes
                "stroke_width": 4,
                "tip_shape": StealthTip,  # Nicer arrow tips
            },
        ).add_coordinates()  # Add coordinate numbers to the axes

        self.play(DrawBorderThenFill(plane), run_time=2)
        self.wait(intro_wait_time)

        # --- 1. Identity Matrix ---
        id_matrix = np.identity(2)  # [[1, 0], [0, 1]]
        # Don't show eigenvectors for identity, they are trivial (any vector is an eigenvector)
        plane = show_transformation(
            "1. Identity Matrix (No Change)", id_matrix, plane, show_eigenvectors=False
        )

        # --- 2. Stretch X-axis (x2) ---
        stretch_x_2 = np.array([[2.0, 0.0], [0.0, 1.0]])
        # Eigenvectors are [1, 0] and [0, 1], eigenvalues are 2 and 1
        plane = show_transformation("2. Stretch X-axis (x2)", stretch_x_2, plane)

        # --- 3. Stretch X-axis (x3) ---
        # We show the *total* transformation from the start (x3)
        # but apply it to the *current* state of the plane (which is already x2 stretched)
        stretch_x_3 = np.array([[3.0, 0.0], [0.0, 1.0]])
        # Eigenvectors are [1, 0] and [0, 1], eigenvalues are 3 and 1
        plane = show_transformation(
            "3. Stretch X-axis (x3)", stretch_x_3, plane
        )  # Shows matrix M=Sx3, applies M to current grid

        # --- Reset Plane for Clarity ---
        self.play(FadeOut(plane))
        plane = NumberPlane(
            x_range=[-7, 7, 1],
            y_range=[-5, 5, 1],
            x_length=14,
            y_length=10,
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 2,
                "stroke_opacity": 0.6,
            },
            axis_config={
                "include_numbers": True,
                "stroke_width": 4,
                "tip_shape": StealthTip,
            },
        ).add_coordinates()
        self.play(FadeIn(plane))
        self.wait(step_wait_time * 0.5)  # Shorter wait after reset

        # --- 4. Stretch Y-axis (x2) ---
        stretch_y_2 = np.array([[1.0, 0.0], [0.0, 2.0]])
        # Eigenvectors are [1, 0] and [0, 1], eigenvalues are 1 and 2
        plane = show_transformation("4. Stretch Y-axis (x2)", stretch_y_2, plane)

        # --- Reset Plane for Clarity ---
        self.play(FadeOut(plane))
        plane = NumberPlane(
            x_range=[-7, 7, 1],
            y_range=[-5, 5, 1],
            x_length=14,
            y_length=10,
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 2,
                "stroke_opacity": 0.6,
            },
            axis_config={
                "include_numbers": True,
                "stroke_width": 4,
                "tip_shape": StealthTip,
            },
        ).add_coordinates()
        self.play(FadeIn(plane))
        self.wait(step_wait_time * 0.5)

        # --- 5. General Symmetric Matrix (Stretch/Shear relative to axes) ---
        # This matrix stretches along non-axis-aligned orthogonal directions.
        # It appears as a combination of stretch and shear from the standard basis perspective.
        symmetric_shear_stretch = np.array([[3.0, 1.0], [1.0, 2.0]])
        # Eigenvalues approx 3.618, 1.382
        # Eigenvectors approx [0.851, 0.526] and [-0.526, 0.851] (orthogonal!)
        plane = show_transformation(
            "5. General Symmetric (Stretch/Shear)", symmetric_shear_stretch, plane
        )

        # --- Reset Plane for Clarity ---
        self.play(FadeOut(plane))
        plane = NumberPlane(
            x_range=[-7, 7, 1],
            y_range=[-5, 5, 1],
            x_length=14,
            y_length=10,
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 2,
                "stroke_opacity": 0.6,
            },
            axis_config={
                "include_numbers": True,
                "stroke_width": 4,
                "tip_shape": StealthTip,
            },
        ).add_coordinates()
        self.play(FadeIn(plane))
        self.wait(step_wait_time * 0.5)

        # --- 6. Symmetric Matrix (Squish/Stretch) ---
        # Construct a matrix designed to squish along one eigenvector and stretch along the other.
        # We choose eigenvalues 3.0 and 0.5.
        # We choose orthogonal eigenvectors [1, 1] and [-1, 1] (normalized).
        eigvals = np.array([3.0, 0.5])
        # Orthonormal eigenvectors: v1 = [1/sqrt(2), 1/sqrt(2)], v2 = [-1/sqrt(2), 1/sqrt(2)]
        eigvecs = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        D = np.diag(eigvals)  # Diagonal matrix of eigenvalues
        # Reconstruct the symmetric matrix: M = V * D * V^T (since V is orthogonal, V_inv = V^T)
        symmetric_squish_stretch = eigvecs @ D @ eigvecs.T
        # This results in matrix: [[1.75, 1.25], [1.25, 1.75]]
        plane = show_transformation(
            "6. Symmetric (Squish/Stretch)", symmetric_squish_stretch, plane
        )

        # --- Final Summary Message ---
        final_text = Text(
            "Key takeaway:\n"
            "Symmetric matrices transform space by scaling along\n"
            "a set of perpendicular directions (their eigenvectors).\n"
            "These directions remain perpendicular after the transformation.",
            font_size=DEFAULT_FONT_SIZE * text_scale,
            line_spacing=1.5,  # Adjust line spacing for readability
        ).center()  # Center the text on the screen

        self.play(FadeOut(plane))  # Fade out the final grid
        self.play(Write(final_text), run_time=3)
        self.wait(5)  # Hold the final message
        self.play(FadeOut(final_text))
        self.wait()  # Short pause at the very end


# To render the scene:
# 1. Save this code as a Python file (e.g., symmetric_transforms.py)
# 2. Open your terminal or command prompt.
# 3. Navigate to the directory where you saved the file.
# 4. Run Manim:
#    manim -pql symmetric_transforms.py SymmetricMatrixTransformations
#
# Flags used:
# -p: Preview the video when rendering is complete.
# -q: Quality flag
#  l: low quality (faster render for testing)
#  m: medium quality
#  h: high quality
#  k: 4k quality
