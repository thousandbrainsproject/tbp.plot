import numpy as np
import logging
import copy
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

def get_relevant_curvature(features):
    """Get the relevant curvature from features. Used to scale search sphere.

    In the case of principal_curvatures and principal_curvatures_log we use the
    maximum absolute curvature between the two values. Otherwise we just return
    the curvature value.

    Note:
        Not sure if other curvatures work as well as log curvatures since they
        may have too big of a range.

    Returns:
        Magnitude of sensed curvature (maximum if using two principal curvatures).
    """
    if "principal_curvatures_log" in features.keys():
        curvatures = features["principal_curvatures_log"]
        curvatures = np.max(np.abs(curvatures))
    elif "principal_curvatures" in features.keys():
        curvatures = features["principal_curvatures"]
        curvatures = np.max(np.abs(curvatures))
    elif "mean_curvature" in features.keys():
        curvatures = features["mean_curvature"]
    elif "mean_curvature_sc" in features.keys():
        curvatures = features["mean_curvature_sc"]
    elif "gaussian_curvature" in features.keys():
        curvatures = features["gaussian_curvature"]
    elif "gaussian_curvature_sc" in features.keys():
        curvatures = features["gaussian_curvature_sc"]
    else:
        logger.error(
            f"No curvatures contained in the features {list(features.keys())}."
        )
        # Return large curvature so we use an almost circular search sphere.
        curvatures = 10
    return curvatures

def get_custom_distances(nearest_node_locs, search_locs, search_sns, search_curvature):
    """Calculate custom distances modulated by surface normal and curvature.

    Args:
        nearest_node_locs: locations of nearest nodes to search_locs.
            shape=(num_hyp, max_nneighbors, 3)
        search_locs: search locations for each hypothesis.
            shape=(num_hyp, 3)
        search_sns: sensed surface normal rotated by hypothesis pose.
            shape=(num_hyp, 3)
        search_curvature: magnitude of sensed curvature (maximum if using
            two principal curvatures). Is used to modulate the search spheres
            thickness in the direction of the surface normal.
            shape=1

    Returns:
        custom_nearest_node_dists: custom distances of each nearest location
            from its search location taking into account the hypothesis point
            normal and sensed curvature.
            shape=(num_hyp, max_nneighbors)
    """
    # Calculate difference vectors between query point and all other points
    # expand_dims of search_locs so it has shape (num_hyp, 1, 3)
    # shape of differences = (num_hyp, max_nneighbors, 3)
    differences = nearest_node_locs - np.expand_dims(search_locs, axis=1)
    # Calculate the dot product between the query normal and the difference vectors
    # This tells us how well the points are aligned with the plane perpendicular to
    # the query normal. Points with dot product 0 are in this plane, higher
    # magnitudes of the dot product means they are further away from that plane
    # (-> should have larger distance).
    dot_products = np.einsum("ijk,ik->ij", differences, search_sns)
    # Calculate the euclidean distances. shape=(num_hyp, max_nneighbors)
    euclidean_dists = np.linalg.norm(differences, axis=2)
    # Calculate the total distances by adding the absolute dot product to the
    # euclidean distances. We multiply the dot product by 1/curvature to modulate
    # the flatness of the search sphere. If the curvature is large we want to be
    # able to go further out of the sphere while we want to stay close to the point
    # normal plane if we have a curvature close to 0.
    # To have a minimum wiggle room above and below the plane, even if we have 0
    # curvature (and to avoid division by 0) we add 0.5 to the denominator.
    # shape=(num_hyp, max_nneighbors).
    custom_nearest_node_dists = euclidean_dists + np.abs(dot_products) * (
        1 / (np.abs(search_curvature) + 0.5)
    )
    return custom_nearest_node_dists

def rotate_pose_dependent_features(features, ref_frame_rots) -> dict:
    """Rotate pose_vectors given a list of rotation matrices.

    Args:
        features: dict of features with pose vectors to rotate.
            pose vectors have shape (3, 3)
        ref_frame_rots: Rotation matrices to rotate pose features by. Can either be
            - A single scipy rotation (as used in FeatureGraphLM)
            - An array of rotation matrices of shape (N, 3, 3) or (3, 3) (as used in
            EvidenceGraphLM).

    Returns:
        Original features but with the pose_vectors rotated. If multiple rotations
        were given, pose_vectors entry will now contain multiple entries of shape
        (N, 3, 3).
    """
    pose_transformed_features = copy.deepcopy(features)
    old_pv = pose_transformed_features["pose_vectors"]
    assert old_pv.shape == (
        3,
        3,
    ), f"pose_vectors in features need to be 3x3 matrices."
    if isinstance(ref_frame_rots, Rotation):
        rotated_pv = ref_frame_rots.apply(old_pv)
    else:
        # Transpose pose vectors so each vector is a column (otherwise .dot matmul
        # produces slightly different results)
        rotated_pv = ref_frame_rots.dot(old_pv.T)
        # Transpose last two axies so each pose vector is a row again
        rotated_pv = rotated_pv.transpose((0, 2, 1))
    pose_transformed_features["pose_vectors"] = rotated_pv
    return pose_transformed_features