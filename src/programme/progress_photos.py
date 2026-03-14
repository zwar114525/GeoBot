"""Progress Photo Integration for construction schedules."""

import os
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


class ProgressPhotoLinker:
    """Link progress photos to schedule activities for verification.

    Uses keyword-based matching to associate site photos with tasks.
    """

    def __init__(self, tasks_df: pd.DataFrame):
        """Initialize with tasks dataframe.

        Expected columns:
        - task_id, task_name: Task identifiers
        - percent_complete: Progress percentage
        - wbs: Work Breakdown Structure
        """
        self.tasks = tasks_df.copy()

        self.activity_keywords = {
            '1000': ['mobilization', 'site office', 'fence', 'setup'],
            '2000': ['excavation', 'digging', 'earthwork', 'grading'],
            '2020': ['foundation', 'footing', 'concrete', 'pad'],
            '3000': ['steel', 'column', 'beam', 'structure', 'frame'],
            '4000': ['wall', 'panel', 'enclosure', 'brick', 'block'],
            '5000': ['roofing', 'roof', 'roofing'],
            '6000': ['mechanical', 'mep', 'hvac', 'plumbing', 'electrical'],
            '7000': ['interior', 'drywall', 'paint', 'finish'],
            '8000': ['paving', 'parking', 'landscape', 'exterior']
        }

    def link_photos_to_activities(self, photo_folder: str) -> Dict:
        """Match site photos to schedule activities using keyword matching.

        Args:
            photo_folder: Path to folder containing photos

        Returns:
            Dictionary mapping task IDs to photo filenames
        """
        if not os.path.exists(photo_folder):
            return {'error': f'Photo folder not found: {photo_folder}'}

        matched_photos = {}

        for filename in os.listdir(photo_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                filename_lower = filename.lower()
                for task_id_prefix, keywords in self.activity_keywords.items():
                    if any(kw in filename_lower for kw in keywords):
                        if task_id_prefix not in matched_photos:
                            matched_photos[task_id_prefix] = []
                        matched_photos[task_id_prefix].append(filename)

        return matched_photos

    def generate_photo_mapping_report(self, photo_folder: str) -> str:
        """Generate a report mapping photos to activities.

        Args:
            photo_folder: Path to folder containing photos

        Returns:
            Formatted report string
        """
        matched = self.link_photos_to_activities(photo_folder)

        if 'error' in matched:
            return matched['error']

        output = "PROGRESS PHOTO MAPPING REPORT\n"
        output += "=" * 60 + "\n\n"

        if not matched:
            return "No photos matched to activities. Ensure photos have descriptive filenames.\n"

        for task_id_prefix, photos in matched.items():
            task = self.tasks[self.tasks['task_id'].astype(str).str.startswith(task_id_prefix)]
            task_name = task.iloc[0]['task_name'] if not task.empty else 'Unknown'
            progress = task.iloc[0]['percent_complete'] if not task.empty else 0

            output += f"[{task_id_prefix}xxx] {task_name}\n"
            output += f"   Progress: {progress}%\n"
            output += f"   Photos: {len(photos)}\n"
            for photo in photos:
                output += f"      - {photo}\n"
            output += "\n"

        return output

    def get_activity_photos(self, task_id: str, photo_folder: str) -> List[str]:
        """Get all photos linked to a specific task.

        Args:
            task_id: Task identifier
            photo_folder: Path to folder containing photos

        Returns:
            List of photo filenames
        """
        matched = self.link_photos_to_activities(photo_folder)

        task_prefix = str(task_id)[:4] if len(str(task_id)) >= 4 else str(task_id)[:2]

        return matched.get(task_prefix, [])
