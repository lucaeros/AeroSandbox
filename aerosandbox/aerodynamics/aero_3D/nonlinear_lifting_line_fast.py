import numpy as onp
import aerosandbox.numpy as np
from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import Airplane
from aerosandbox.performance import OperatingPoint
from aerosandbox.aerodynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import (
    calculate_induced_velocity_horseshoe,
)
from aerosandbox.modeling.interpolation import InterpolatedModel
import aerosandbox.geometry.mesh_utilities as mesh_utils
from typing import Any, Callable, Dict, List, TYPE_CHECKING
from scipy import optimize

if TYPE_CHECKING:
    from aerosandbox.geometry import Airfoil, ControlSurface


def tall(array):
    """Reshape an array into a tall (Nx1) column vector."""
    return np.reshape(array, (-1, 1))


def wide(array):
    """Reshape an array into a wide (1xN) row vector."""
    return np.reshape(array, (1, -1))


class NonlinearLiftingLineFast(ExplicitAnalysis):
    """
    A fast nonlinear lifting-line aerodynamics analysis.

    Same physics as `NonlinearLiftingLine`, but replaces the per-run CasADi/IPOPT solve of
    the symbolic NeuralFoil graph with a two-stage scheme (mirroring the `compute_polars`
    trick of `ViscousVortexLatticeMethod`):

    1. `compute_polars()` evaluates each spanwise section's ``CL/CD/CM(Re, alpha)`` once with a
       single batched NeuralFoil call per section, and stores cheap `InterpolatedModel`s.
    2. `run()` solves the ``n_panels`` circulation unknowns with a numeric Newton iteration
       against those interpolants (no NeuralFoil evaluation, no CasADi NLP construction).

    Because the section polars and the panel geometry are independent of both angle of attack
    and circulation (Reynolds number and Mach are set by the freestream speed), they are
    computed once and reused across an entire alpha sweep. In practice this is ~1-2 orders of
    magnitude faster than `NonlinearLiftingLine` per operating point, and the circulation solve
    can be warm-started across a sweep via `vortex_strengths_init`.

    Examples
    --------
    >>> analysis = asb.NonlinearLiftingLineFast(
    >>>     airplane=my_airplane,
    >>>     op_point=asb.OperatingPoint(velocity=25, alpha=5),
    >>> )
    >>> aero = analysis.run()
    """

    def __init__(
        self,
        airplane: Airplane,
        op_point: OperatingPoint,
        xyz_ref: List[float] | None = None,
        run_symmetric_if_possible: bool = False,
        verbose: bool = False,
        spanwise_resolution: int = 8,
        spanwise_spacing_function: Callable[
            [float, float, int], np.ndarray
        ] = np.cosspace,
        vortex_core_radius: float = 1e-8,
        align_trailing_vortices_with_wind: bool = False,
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        model_size: str = "large",
    ):
        """
        Initialize a NonlinearLiftingLineFast analysis.

        Parameters
        ----------
        airplane : Airplane
            The Airplane object to analyze.
        op_point : OperatingPoint
            The OperatingPoint to analyze the airplane at.
        xyz_ref : list[float] | None
            The moment reference point, in geometry axes. Defaults to `airplane.xyz_ref`.
        run_symmetric_if_possible : bool
            Not implemented; kept for signature parity with the other solvers.
        verbose : bool
            If True, prints progress messages during the analysis.
        spanwise_resolution : int
            The number of spanwise panels that each wing section is subdivided into.
        spanwise_spacing_function : Callable[[float, float, int], np.ndarray]
            Spanwise panel spacing function (e.g., `np.linspace` or `np.cosspace`).
        vortex_core_radius : float
            The regularization radius of each vortex core [m].
        align_trailing_vortices_with_wind : bool
            If True, trailing vortex legs align with the freestream direction; else +x
            (geometry axes).
        n_crit, xtr_upper, xtr_lower : float
            NeuralFoil transition parameters used when precomputing the section polars.
        model_size : str
            NeuralFoil model size used when precomputing the section polars.
        """
        super().__init__()

        if xyz_ref is None:
            xyz_ref = airplane.xyz_ref

        self.airplane = airplane
        self.op_point = op_point
        self.xyz_ref = xyz_ref
        self.verbose = verbose
        self.spanwise_resolution = spanwise_resolution
        self.spanwise_spacing_function = spanwise_spacing_function
        self.vortex_core_radius = vortex_core_radius
        self.align_trailing_vortices_with_wind = align_trailing_vortices_with_wind
        self.n_crit = n_crit
        self.xtr_lower = xtr_lower
        self.xtr_upper = xtr_upper
        self.model_size = model_size

        self._meshed = False
        self._section_polars = None
        self._polar_meta = None
        self._polar_stacks = None
        self._polar_axes = None

        self.run_symmetric = False
        if run_symmetric_if_possible:
            raise NotImplementedError(
                "NonlinearLiftingLineFast with symmetry detection not yet implemented!"
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(\n\t"
            + "\n\t".join(
                [
                    f"airplane={self.airplane}",
                    f"op_point={self.op_point}",
                    f"xyz_ref={self.xyz_ref}",
                ]
            )
            + "\n)"
        )

    ### ----------------------------------------------------------------- Meshing

    def _mesh(self) -> None:
        """Build the horseshoe-panel geometry (alpha/circulation independent). Mirrors the
        meshing block of `NonlinearLiftingLine.run()`."""
        if self._meshed:
            return

        front_left_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        front_right_vertices = []
        airfoils: List["Airfoil"] = []
        control_surfaces: List[List["ControlSurface"]] = []

        for wing in self.airplane.wings:
            if self.spanwise_resolution > 1:
                wing = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function,
                )

            points, faces = wing.mesh_thin_surface(
                method="quad",
                chordwise_resolution=1,
                add_camber=False,
            )

            # place middle section to zero
            points[faces[:1, :2], 1] = 0
            points[
                faces[
                    faces.shape[0] // 2 : faces.shape[0] // 2 + 1,
                    2:4,
                ],
                1,
            ] = 0

            front_left_vertices.append(points[faces[:, 0], :])
            back_left_vertices.append(points[faces[:, 1], :])
            back_right_vertices.append(points[faces[:, 2], :])
            front_right_vertices.append(points[faces[:, 3], :])

            wing_airfoils = []
            wing_control_surfaces = []
            for xsec_a, xsec_b in zip(wing.xsecs[:-1], wing.xsecs[1:]):
                wing_airfoils.append(
                    xsec_a.airfoil.blend_with_another_airfoil(
                        airfoil=xsec_b.airfoil,
                        blend_fraction=0.5,
                    )
                )
                wing_control_surfaces.append(xsec_a.control_surfaces)

            airfoils.extend(wing_airfoils)
            control_surfaces.extend(wing_control_surfaces)

            if wing.symmetric:
                airfoils.extend(wing_airfoils)

                def mirror_control_surface(surf: "ControlSurface") -> "ControlSurface":
                    if surf.symmetric:
                        return surf
                    else:
                        surf = surf.copy()
                        surf.deflection *= -1
                        return surf

                symmetric_wing_control_surfaces = [
                    [mirror_control_surface(surf) for surf in surfs]
                    for surfs in wing_control_surfaces
                ]
                control_surfaces.extend(symmetric_wing_control_surfaces)

        front_left_vertices = np.concatenate(front_left_vertices)
        back_left_vertices = np.concatenate(back_left_vertices)
        back_right_vertices = np.concatenate(back_right_vertices)
        front_right_vertices = np.concatenate(front_right_vertices)

        ### Panel statistics
        diag1 = front_right_vertices - back_left_vertices
        diag2 = front_left_vertices - back_right_vertices
        cross = np.cross(diag1, diag2)
        cross_norm = np.linalg.norm(cross, axis=1)
        normal_directions = cross / tall(cross_norm)
        areas = cross_norm / 2

        left_vortex_vertices = 0.75 * front_left_vertices + 0.25 * back_left_vertices
        right_vortex_vertices = 0.75 * front_right_vertices + 0.25 * back_right_vertices
        vortex_centers = (left_vortex_vertices + right_vortex_vertices) / 2
        vortex_bound_leg = right_vortex_vertices - left_vortex_vertices
        vortex_bound_leg_norm = np.linalg.norm(vortex_bound_leg, axis=1)
        chord_vectors = (back_left_vertices + back_right_vertices) / 2 - (
            front_left_vertices + front_right_vertices
        ) / 2
        chords = np.linalg.norm(chord_vectors, axis=1)
        wing_directions = vortex_bound_leg / tall(vortex_bound_leg_norm)
        local_forward_direction = np.cross(normal_directions, wing_directions)

        self.front_left_vertices = front_left_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.front_right_vertices = front_right_vertices
        self.airfoils = airfoils
        self.control_surfaces = control_surfaces
        self.normal_directions = normal_directions
        self.areas = areas
        self.left_vortex_vertices = left_vortex_vertices
        self.right_vortex_vertices = right_vortex_vertices
        self.vortex_centers = vortex_centers
        self.vortex_bound_leg = vortex_bound_leg
        self.chord_vectors = chord_vectors
        self.chords = chords
        self.local_forward_direction = local_forward_direction
        self.n_panels = areas.shape[0]

        self._meshed = True

    def _update_freestream(self) -> None:
        """Recompute the (op-point-dependent) freestream vectors for the current op_point."""
        v = self.op_point.compute_freestream_velocity_geometry_axes()
        self.steady_freestream_velocity = v
        self.steady_freestream_direction = v / np.linalg.norm(v)

    ### ---------------------------------------------------------- Section polars

    def compute_polars(
        self,
        velocities: np.ndarray = None,
        alphas: np.ndarray = None,
        n_Re: int = 6,
        Re_bounds: tuple = None,
        mach: float = None,
        model_size: str = None,
    ) -> None:
        """Precompute a per-section ``CL/CD/CM(Re, alpha)`` interpolant so `run()` needs no
        NeuralFoil calls.

        Section polars depend only on Reynolds number and angle of attack (fixed geometry),
        so precomputing once lets you sweep alpha / drive a trim loop very cheaply.

        Parameters
        ----------
        velocities : np.ndarray
            Freestream speeds [m/s] the Re grid must cover (uses section chords). If None,
            brackets the current op_point's per-section Reynolds numbers.
        alphas : np.ndarray
            AoA samples [deg]. Default 100 points over [-15, 25].
        n_Re : int
            Reynolds grid size.
        Re_bounds : tuple
            Explicit (Re_min, Re_max); overrides `velocities` if given.
        mach : float
            Mach for the polar; defaults to the operating point's Mach.
        model_size : str
            NeuralFoil model size; defaults to `self.model_size`.
        """
        self._mesh()

        if alphas is None:
            alphas = np.linspace(-15, 25, 100)
        if model_size is None:
            model_size = self.model_size
        if mach is None:
            mach = self.op_point.mach()

        chords = self.chords

        if Re_bounds is None:
            if velocities is not None:
                velocities = np.atleast_1d(np.array(velocities, dtype=float))
                mu = self.op_point.atmosphere.dynamic_viscosity()
                rho = self.op_point.atmosphere.density()
                Re_bounds = (
                    rho * np.min(velocities) * np.min(chords) / mu,
                    rho * np.max(velocities) * np.max(chords) / mu,
                )
            else:
                Re_sections = self.op_point.reynolds(chords)
                Re_bounds = (0.4 * np.min(Re_sections), 2.0 * np.max(Re_sections))

        Re_grid = np.geomspace(Re_bounds[0], Re_bounds[1], n_Re)

        cl_polars = []
        cd_polars = []
        cm_polars = []
        cl_data = []
        cd_data = []
        cm_data = []
        for i, af in enumerate(self.airfoils):
            CL_grid = np.empty((n_Re, len(alphas)))
            CD_grid = np.empty((n_Re, len(alphas)))
            CM_grid = np.empty((n_Re, len(alphas)))
            for k, Re in enumerate(Re_grid):
                aero = af.get_aero_from_neuralfoil(
                    alpha=alphas,
                    Re=float(Re),
                    mach=mach,
                    control_surfaces=self.control_surfaces[i],
                    xtr_lower=self.xtr_lower,
                    xtr_upper=self.xtr_upper,
                    n_crit=self.n_crit,
                    model_size=model_size,
                )
                CL_grid[k, :] = np.array(aero["CL"]).flatten()
                CD_grid[k, :] = np.array(aero["CD"]).flatten()
                CM_grid[k, :] = np.array(aero["CM"]).flatten()

            grid = {"Re": Re_grid, "alpha": alphas}
            cl_polars.append(
                InterpolatedModel(grid, CL_grid, method="linear", fill_value=None)
            )
            cd_polars.append(
                InterpolatedModel(grid, CD_grid, method="linear", fill_value=None)
            )
            cm_polars.append(
                InterpolatedModel(grid, CM_grid, method="linear", fill_value=None)
            )
            cl_data.append(CL_grid)
            cd_data.append(CD_grid)
            cm_data.append(CM_grid)

        self._section_polars = {"CL": cl_polars, "CD": cd_polars, "CM": cm_polars}
        # Stacked (n_sections, n_Re, n_alpha) arrays over the shared axes, for the
        # vectorized `_eval_section` batched interpolation path.
        self._polar_axes = (
            onp.asarray(Re_grid, dtype=float),
            onp.asarray(alphas, dtype=float),
        )
        self._polar_stacks = {
            "CL": onp.asarray(cl_data, dtype=float),
            "CD": onp.asarray(cd_data, dtype=float),
            "CM": onp.asarray(cm_data, dtype=float),
        }
        self._polar_meta = {
            "Re_grid": Re_grid,
            "alphas": alphas,
            "n_sections": len(self.airfoils),
        }

    def _eval_section(self, key: str, Res, alphas) -> np.ndarray:
        """Evaluate every panel's section coefficient at its own (Re, alpha).

        All sections share the same (Re, alpha) grid axes, so this is a single vectorized
        bilinear interpolation (with scipy-style linear extrapolation) over the stacked polar
        data, instead of one scalar `interpn` call per panel.
        """
        if getattr(self, "_polar_stacks", None) is None:
            polars = self._section_polars[key]
            return np.array(
                [
                    float(polars[i]({"Re": Res[i], "alpha": alphas[i]}))
                    for i in range(self.n_panels)
                ]
            )

        Re_grid, alpha_grid = self._polar_axes
        data = self._polar_stacks[key]  # (n_panels, n_Re, n_alpha)

        Res = onp.asarray(Res, dtype=float)
        alphas = onp.asarray(alphas, dtype=float)

        # scipy RegularGridInterpolator-style index/fraction (fraction goes <0 or >1 to
        # extrapolate, matching fill_value=None on the InterpolatedModels).
        iR = onp.clip(onp.searchsorted(Re_grid, Res) - 1, 0, Re_grid.size - 2)
        fR = (Res - Re_grid[iR]) / (Re_grid[iR + 1] - Re_grid[iR])
        iA = onp.clip(onp.searchsorted(alpha_grid, alphas) - 1, 0, alpha_grid.size - 2)
        fA = (alphas - alpha_grid[iA]) / (alpha_grid[iA + 1] - alpha_grid[iA])

        p = onp.arange(self.n_panels)
        c00 = data[p, iR, iA]
        c10 = data[p, iR + 1, iA]
        c01 = data[p, iR, iA + 1]
        c11 = data[p, iR + 1, iA + 1]
        return (
            c00 * (1 - fR) * (1 - fA)
            + c10 * fR * (1 - fA)
            + c01 * (1 - fR) * fA
            + c11 * fR * fA
        )

    ### ---------------------------------------------------------- Velocity field

    def get_induced_velocity_at_points(
        self, points: np.ndarray, vortex_strengths: np.ndarray = None
    ) -> np.ndarray:
        """Nx3 induced velocity at `points` (geometry axes) from all horseshoes."""
        if vortex_strengths is None:
            vortex_strengths = self.vortex_strengths
        u_induced, v_induced, w_induced = calculate_induced_velocity_horseshoe(
            x_field=tall(points[:, 0]),
            y_field=tall(points[:, 1]),
            z_field=tall(points[:, 2]),
            x_left=wide(self.left_vortex_vertices[:, 0]),
            y_left=wide(self.left_vortex_vertices[:, 1]),
            z_left=wide(self.left_vortex_vertices[:, 2]),
            x_right=wide(self.right_vortex_vertices[:, 0]),
            y_right=wide(self.right_vortex_vertices[:, 1]),
            z_right=wide(self.right_vortex_vertices[:, 2]),
            trailing_vortex_direction=(
                self.steady_freestream_direction
                if self.align_trailing_vortices_with_wind
                else np.array([1, 0, 0])
            ),
            gamma=wide(vortex_strengths),
            vortex_core_radius=self.vortex_core_radius,
        )
        u_induced = np.sum(u_induced, axis=1)
        v_induced = np.sum(v_induced, axis=1)
        w_induced = np.sum(w_induced, axis=1)
        return np.stack([u_induced, v_induced, w_induced], axis=1)

    def get_velocity_at_points(
        self, points: np.ndarray, vortex_strengths: np.ndarray = None
    ) -> np.ndarray:
        """Nx3 total velocity at `points` (geometry axes)."""
        V_induced = self.get_induced_velocity_at_points(points, vortex_strengths)
        rotation_freestream_velocities = np.array(
            self.op_point.compute_rotation_velocity_geometry_axes(points)
        )
        freestream_velocities = np.add(
            wide(self.steady_freestream_velocity), rotation_freestream_velocities
        )
        return V_induced + freestream_velocities

    ### ------------------------------------------------------------- Streamlines

    def calculate_streamlines(
        self,
        seed_points: np.ndarray | None = None,
        n_steps: int = 300,
        length: float | None = None,
    ) -> np.ndarray:
        """Trace streamlines by forward-Euler integration of the velocity field. Result is
        stored as `self.streamlines` (n_seed x 3 x n_steps)."""
        if length is None:
            length = self.airplane.c_ref * 5
        if seed_points is None:
            left_TE_vertices = self.back_left_vertices
            right_TE_vertices = self.back_right_vertices
            seed_points_per_panel = np.maximum(1, 200 // len(left_TE_vertices))
            nondim_node_locations = np.linspace(0, 1, seed_points_per_panel + 1)
            nondim_seed_locations = (
                nondim_node_locations[1:] + nondim_node_locations[:-1]
            ) / 2
            seed_points = np.concatenate(
                [
                    x * left_TE_vertices + (1 - x) * right_TE_vertices
                    for x in nondim_seed_locations
                ]
            )

        streamlines = np.empty((len(seed_points), 3, n_steps))
        streamlines[:, :, 0] = seed_points
        for i in range(1, n_steps):
            V = self.get_velocity_at_points(streamlines[:, :, i - 1])
            streamlines[:, :, i] = streamlines[
                :, :, i - 1
            ] + length / n_steps * V / tall(np.linalg.norm(V, axis=1))
        self.streamlines = streamlines
        return streamlines

    def draw(
        self,
        c: np.ndarray | None = None,
        cmap: str | None = None,
        colorbar_label: str | None = None,
        show: bool = True,
        show_kwargs: dict | None = None,
        draw_streamlines=True,
        recalculate_streamlines=False,
        backend: str = "pyvista",
    ):
        """Draw the solved circulation distribution (and optional streamlines). Must be called
        after `run()`."""
        if show_kwargs is None:
            show_kwargs = {}

        if c is None:
            c = self.vortex_strengths
            colorbar_label = "Vortex Strengths"

        if draw_streamlines:
            if (not hasattr(self, "streamlines")) or recalculate_streamlines:
                self.calculate_streamlines()

        if backend == "plotly":
            from aerosandbox.visualization.plotly_Figure3D import Figure3D

            fig = Figure3D()
            for i in range(len(self.front_left_vertices)):
                fig.add_quad(
                    points=[
                        self.front_left_vertices[i, :],
                        self.back_left_vertices[i, :],
                        self.back_right_vertices[i, :],
                        self.front_right_vertices[i, :],
                    ],
                    intensity=c[i],
                    outline=True,
                )
            if draw_streamlines:
                for i in range(self.streamlines.shape[0]):
                    fig.add_streamline(self.streamlines[i, :, :].T)
            return fig.draw(show=show, colorbar_title=colorbar_label, **show_kwargs)

        elif backend == "pyvista":
            import pyvista as pv

            plotter = pv.Plotter()
            plotter.title = "ASB NonlinearLiftingLineFast"
            plotter.add_axes()
            plotter.show_grid(color="gray")

            points = np.concatenate(
                [
                    self.front_left_vertices,
                    self.back_left_vertices,
                    self.back_right_vertices,
                    self.front_right_vertices,
                ]
            )
            N = len(self.front_left_vertices)
            range_N = np.arange(N)
            faces = tall(range_N) + wide(np.array([0, 1, 2, 3]) * N)
            mesh = pv.PolyData(
                *mesh_utils.convert_mesh_to_polydata_format(points, faces)
            )
            scalar_bar_args = {}
            if colorbar_label is not None:
                scalar_bar_args["title"] = colorbar_label
            plotter.add_mesh(
                mesh=mesh,
                scalars=c,
                show_edges=True,
                show_scalar_bar=c is not None,
                scalar_bar_args=scalar_bar_args,
                cmap=cmap,
            )
            if draw_streamlines:
                import aerosandbox.tools.pretty_plots as p

                for i in range(self.streamlines.shape[0]):
                    plotter.add_mesh(
                        pv.Spline(self.streamlines[i, :, :].T),
                        color=p.adjust_lightness("#7700FF", 1.5),
                        opacity=0.7,
                        line_width=1,
                    )
            if show:
                plotter.show(**show_kwargs)
            return plotter

        else:
            raise ValueError("Bad value of `backend`!")

    ### ------------------------------------------------------------- Solve + run

    def _panel_kinematics(self, vortex_strengths: np.ndarray):
        velocities = self.get_velocity_at_points(self.vortex_centers, vortex_strengths)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        velocity_directions = velocities / tall(velocity_magnitudes)
        alphas = 90 - np.arccosd(
            np.sum(velocity_directions * self.normal_directions, axis=1)
        )
        cos_sweeps = np.sum(velocity_directions * -self.local_forward_direction, axis=1)
        Res = (
            velocity_magnitudes
            * self.chords
            / self.op_point.atmosphere.kinematic_viscosity()
        ) * cos_sweeps
        return velocities, velocity_magnitudes, alphas, cos_sweeps, Res

    def _residuals(self, vortex_strengths: np.ndarray) -> np.ndarray:
        vortex_strengths = np.asarray(vortex_strengths, dtype=float)
        velocities, velocity_magnitudes, alphas, cos_sweeps, Res = (
            self._panel_kinematics(vortex_strengths)
        )
        CLs = self._eval_section("CL", Res, alphas)
        Vi_cross_li = np.cross(velocities, self.vortex_bound_leg, axis=1)
        Vi_cross_li_magnitudes = np.linalg.norm(Vi_cross_li, axis=1)
        velocity_magnitude_perpendiculars = velocity_magnitudes * cos_sweeps
        return (
            vortex_strengths
            * Vi_cross_li_magnitudes
            * 2
            / velocity_magnitude_perpendiculars**2
            / self.areas
            - CLs
        )

    def run(
        self,
        alpha: float = None,
        vortex_strengths_init: np.ndarray = None,
        tol: float = 1e-9,
    ) -> Dict[str, Any]:
        """
        Solve for the aerodynamic forces and moments at the current operating point.

        Parameters
        ----------
        alpha : float
            If given, overrides the operating point's angle of attack [deg] for this run.
            Useful for sweeping alpha on a single, already-meshed instance.
        vortex_strengths_init : np.ndarray
            Initial guess for the circulation solve. Pass the previous solution to warm-start
            an alpha sweep.
        tol : float
            Convergence tolerance for the Newton (root-find) solve.

        Returns
        -------
        dict[str, Any]
            The same keys as `NonlinearLiftingLine.run()`: forces/moments in geometry, body,
            and wind axes, plus CL, CD, CDi, CDp, CY, Cl, Cm, Cn, and `vortex_strengths`.
        """
        self._mesh()
        if alpha is not None:
            self.op_point.alpha = alpha
        self._update_freestream()

        if self._section_polars is None:
            self.compute_polars()

        ##### Solve for circulation (numeric Newton root-find)
        if vortex_strengths_init is None:
            vortex_strengths_init = np.zeros(self.n_panels)
        sol = optimize.root(
            self._residuals,
            np.asarray(vortex_strengths_init, dtype=float),
            method="hybr",
            tol=tol,
        )
        vortex_strengths = sol.x
        self.vortex_strengths = vortex_strengths
        self.solve_success = bool(sol.success)

        ##### Evaluate converged panel state
        velocities, velocity_magnitudes, alphas, cos_sweeps, Res = (
            self._panel_kinematics(vortex_strengths)
        )
        CDs = self._eval_section("CD", Res, alphas)
        CMs = self._eval_section("CM", Res, alphas)
        self.CLs = self._eval_section("CL", Res, alphas)
        self.CDs = CDs
        self.CMs = CMs
        self.alphas = alphas

        ##### Forces (identical formulation to NonlinearLiftingLine)
        rho = self.op_point.atmosphere.density()

        Vi_cross_li = np.cross(velocities, self.vortex_bound_leg, axis=1)
        forces_inviscid_geometry = rho * Vi_cross_li * tall(vortex_strengths)
        moments_inviscid_geometry = np.cross(
            np.add(self.vortex_centers, -wide(np.array(self.xyz_ref))),
            forces_inviscid_geometry,
        )
        force_inviscid_geometry = np.sum(forces_inviscid_geometry, axis=0)

        forces_profile_geometry = (
            0.5
            * rho
            * velocities
            * tall(velocity_magnitudes)
            * tall(CDs)
            * tall(self.areas)
        )
        moments_profile_geometry = np.cross(
            np.add(self.vortex_centers, -wide(np.array(self.xyz_ref))),
            forces_profile_geometry,
        )
        force_profile_geometry = np.sum(forces_profile_geometry, axis=0)

        bound_leg_YZ = np.concatenate(
            [np.zeros((self.n_panels, 1)), self.vortex_bound_leg[:, 1:]], axis=1
        )
        moments_pitching_geometry = (
            (0.5 * rho * tall(velocity_magnitudes**2))
            * tall(CMs)
            * tall(self.chords**2)
            * bound_leg_YZ
        )

        ### To body & wind axes
        def _convert(vec, from_axes, to_axes):
            return np.array(
                self.op_point.convert_axes(
                    vec[0], vec[1], vec[2], from_axes=from_axes, to_axes=to_axes
                )
            )

        force_inviscid_body = _convert(force_inviscid_geometry, "geometry", "body")
        force_inviscid_wind = _convert(force_inviscid_body, "body", "wind")
        force_profile_body = _convert(force_profile_geometry, "geometry", "body")
        force_profile_wind = _convert(force_profile_body, "body", "wind")

        moment_inviscid_geometry = np.sum(moments_inviscid_geometry, axis=0)
        moment_profile_geometry = np.sum(moments_profile_geometry, axis=0)
        moment_pitching_geometry = np.sum(moments_pitching_geometry, axis=0)

        force_total_geometry = np.add(force_inviscid_geometry, force_profile_geometry)
        force_total_body = _convert(force_total_geometry, "geometry", "body")
        force_total_wind = _convert(force_total_body, "body", "wind")

        moment_total_geometry = (
            np.add(moment_inviscid_geometry, moment_profile_geometry)
            + moment_pitching_geometry
        )
        moment_total_body = _convert(moment_total_geometry, "geometry", "body")
        moment_total_wind = _convert(moment_total_body, "body", "wind")

        L = -force_total_wind[2]
        D = -force_total_wind[0]
        Di = -force_inviscid_wind[0]
        Dp = -force_profile_wind[0]
        Y = force_total_wind[1]
        l_b = moment_total_body[0]
        m_b = moment_total_body[1]
        n_b = moment_total_body[2]

        q = self.op_point.dynamic_pressure()
        s_ref = self.airplane.s_ref
        b_ref = self.airplane.b_ref
        c_ref = self.airplane.c_ref

        self.CL = L / q / s_ref
        self.CD = D / q / s_ref
        CDi = Di / q / s_ref
        CDp = Dp / q / s_ref
        self.CY = Y / q / s_ref
        self.Cl = l_b / q / s_ref / b_ref
        self.Cm = m_b / q / s_ref / c_ref
        self.Cn = n_b / q / s_ref / b_ref
        self.CL_over_CD = np.where(self.CD == 0, 0, np.array(self.CL / self.CD))

        return {
            "F_g": force_total_geometry,
            "F_b": force_total_body,
            "F_w": force_total_wind,
            "M_g": moment_total_geometry,
            "M_b": moment_total_body,
            "M_w": moment_total_wind,
            "L": L,
            "D": D,
            "Y": Y,
            "l_b": l_b,
            "m_b": m_b,
            "n_b": n_b,
            "CL": self.CL,
            "CD": self.CD,
            "CDi": CDi,
            "CDp": CDp,
            "CY": self.CY,
            "CL_over_CD": self.CL_over_CD,
            "Cl": self.Cl,
            "Cm": self.Cm,
            "Cn": self.Cn,
            "vortex_strengths": vortex_strengths,
        }

    def solve_cl(
        self,
        CL_target: float,
        alpha_bracket: tuple = (-10.0, 17.0),
        maxiter: int = 10,
    ) -> Dict[str, Any]:
        """Trim to a target lift coefficient by solving for angle of attack [deg].

        Reuses the cached mesh and precomputed section polars across iterations, and warm-starts
        the circulation solve from the previous alpha, so only the alpha-dependent solve is
        repeated. Sets `op_point.alpha` to the solution and returns the converged `run()` result
        dict with an added 'alpha' key.
        """
        self._mesh()
        if self._section_polars is None:
            self.compute_polars()

        vortex_strengths_init = None

        def residual(alpha):
            nonlocal vortex_strengths_init
            result = self.run(alpha=alpha, vortex_strengths_init=vortex_strengths_init)
            vortex_strengths_init = result["vortex_strengths"]
            return result["CL"] - CL_target

        sol = optimize.root_scalar(
            residual, bracket=list(alpha_bracket), method="brentq", maxiter=maxiter
        )
        result = self.run(alpha=sol.root, vortex_strengths_init=vortex_strengths_init)
        result["alpha"] = sol.root
        return result
