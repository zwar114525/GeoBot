"""XER Parser - Parse Primavera P6 XER files for construction schedules."""

import pandas as pd
from datetime import datetime, timedelta
import os
import re


class XERParser:
    """Parse and process Primavera P6 XER schedule files."""

    def __init__(self, file_path: str = None, use_simulated: bool = False):
        self.file_path = file_path
        self.use_simulated = use_simulated
        self.tasks = pd.DataFrame()
        self.resources = pd.DataFrame()
        self.relationships = pd.DataFrame()

        if use_simulated or file_path is None:
            self._load_simulated_xer()
        elif os.path.exists(file_path):
            self._parse_real_xer()
        else:
            raise FileNotFoundError(f"XER file not found: {file_path}")

    def _load_simulated_xer(self):
        """Generate realistic construction schedule data for testing."""
        base_date = datetime(2024, 1, 1)

        activities = [
            {"task_id": "1000", "task_name": "Project Mobilization", "wbs": "1.0",
             "start": 0, "duration": 10, "complete": 100, "cost": 50000, "critical": True, "crew": "Mobilization"},
            {"task_id": "1010", "task_name": "Site Survey & Layout", "wbs": "1.0",
             "start": 10, "duration": 5, "complete": 100, "cost": 25000, "critical": True, "crew": "Survey"},
            {"task_id": "1020", "task_name": "Site Clearing & Grubbing", "wbs": "1.0",
             "start": 15, "duration": 10, "complete": 100, "cost": 75000, "critical": True, "crew": "Civil"},

            {"task_id": "2000", "task_name": "Excavation - Zone A", "wbs": "2.0",
             "start": 25, "duration": 15, "complete": 100, "cost": 150000, "critical": True, "crew": "Civil"},
            {"task_id": "2010", "task_name": "Excavation - Zone B", "wbs": "2.0",
             "start": 35, "duration": 12, "complete": 80, "cost": 120000, "critical": False, "crew": "Civil"},
            {"task_id": "2020", "task_name": "Foundation Footings - Zone A", "wbs": "2.0",
             "start": 40, "duration": 20, "complete": 90, "cost": 280000, "critical": True, "crew": "Concrete"},
            {"task_id": "2030", "task_name": "Foundation Footings - Zone B", "wbs": "2.0",
             "start": 47, "duration": 18, "complete": 60, "cost": 250000, "critical": False, "crew": "Concrete"},
            {"task_id": "2040", "task_name": "Foundation Walls - Zone A", "wbs": "2.0",
             "start": 60, "duration": 15, "complete": 70, "cost": 320000, "critical": True, "crew": "Concrete"},
            {"task_id": "2050", "task_name": "Foundation Walls - Zone B", "wbs": "2.0",
             "start": 65, "duration": 14, "complete": 40, "cost": 290000, "critical": False, "crew": "Concrete"},
            {"task_id": "2060", "task_name": "Waterproofing & Backfill", "wbs": "2.0",
             "start": 75, "duration": 10, "complete": 30, "cost": 95000, "critical": True, "crew": "Civil"},

            {"task_id": "3000", "task_name": "Steel Columns - Zone A", "wbs": "3.0",
             "start": 75, "duration": 20, "complete": 50, "cost": 450000, "critical": True, "crew": "Steel"},
            {"task_id": "3010", "task_name": "Steel Columns - Zone B", "wbs": "3.0",
             "start": 80, "duration": 18, "complete": 35, "cost": 420000, "critical": False, "crew": "Steel"},
            {"task_id": "3020", "task_name": "Steel Beams - Level 1", "wbs": "3.0",
             "start": 95, "duration": 25, "complete": 30, "cost": 580000, "critical": True, "crew": "Steel"},
            {"task_id": "3030", "task_name": "Steel Beams - Level 2", "wbs": "3.0",
             "start": 110, "duration": 22, "complete": 10, "cost": 550000, "critical": True, "crew": "Steel"},
            {"task_id": "3040", "task_name": "Metal Decking Installation", "wbs": "3.0",
             "start": 125, "duration": 20, "complete": 5, "cost": 380000, "critical": True, "crew": "Steel"},

            {"task_id": "4000", "task_name": "Exterior Wall Panels - Zone A", "wbs": "4.0",
             "start": 130, "duration": 30, "complete": 0, "cost": 420000, "critical": True, "crew": "Enclosure"},
            {"task_id": "4010", "task_name": "Exterior Wall Panels - Zone B", "wbs": "4.0",
             "start": 135, "duration": 28, "complete": 0, "cost": 390000, "critical": False, "crew": "Enclosure"},
            {"task_id": "4020", "task_name": "Roofing System Installation", "wbs": "4.0",
             "start": 145, "duration": 25, "complete": 0, "cost": 350000, "critical": True, "crew": "Roofing"},
            {"task_id": "4030", "task_name": "Window & Glazing Installation", "wbs": "4.0",
             "start": 155, "duration": 30, "complete": 0, "cost": 280000, "critical": False, "crew": "Glazing"},

            {"task_id": "5000", "task_name": "HVAC Ductwork Installation", "wbs": "5.0",
             "start": 140, "duration": 35, "complete": 0, "cost": 520000, "critical": True, "crew": "MEP"},
            {"task_id": "5010", "task_name": "Electrical Rough-In", "wbs": "5.0",
             "start": 145, "duration": 30, "complete": 0, "cost": 480000, "critical": False, "crew": "Electrical"},
            {"task_id": "5020", "task_name": "Plumbing Rough-In", "wbs": "5.0",
             "start": 145, "duration": 28, "complete": 0, "cost": 390000, "critical": False, "crew": "Plumbing"},
            {"task_id": "5030", "task_name": "Fire Protection System", "wbs": "5.0",
             "start": 160, "duration": 25, "complete": 0, "cost": 310000, "critical": True, "crew": "Fire"},

            {"task_id": "6000", "task_name": "Drywall Installation", "wbs": "6.0",
             "start": 170, "duration": 35, "complete": 0, "cost": 290000, "critical": True, "crew": "Finishing"},
            {"task_id": "6010", "task_name": "Painting & Wall Finishes", "wbs": "6.0",
             "start": 190, "duration": 25, "complete": 0, "cost": 180000, "critical": False, "crew": "Finishing"},
            {"task_id": "6020", "task_name": "Flooring Installation", "wbs": "6.0",
             "start": 200, "duration": 30, "complete": 0, "cost": 320000, "critical": True, "crew": "Finishing"},
            {"task_id": "6030", "task_name": "Ceiling Grid & Tiles", "wbs": "6.0",
             "start": 195, "duration": 20, "complete": 0, "cost": 150000, "critical": False, "crew": "Finishing"},

            {"task_id": "7000", "task_name": "Site Landscaping", "wbs": "7.0",
             "start": 210, "duration": 25, "complete": 0, "cost": 180000, "critical": False, "crew": "Landscaping"},
            {"task_id": "7010", "task_name": "Parking Lot Paving", "wbs": "7.0",
             "start": 215, "duration": 20, "complete": 0, "cost": 220000, "critical": True, "crew": "Civil"},
            {"task_id": "7020", "task_name": "Site Lighting & Signage", "wbs": "7.0",
             "start": 225, "duration": 15, "complete": 0, "cost": 95000, "critical": False, "crew": "Electrical"},

            {"task_id": "8000", "task_name": "System Commissioning", "wbs": "8.0",
             "start": 230, "duration": 20, "complete": 0, "cost": 150000, "critical": True, "crew": "Commissioning"},
            {"task_id": "8010", "task_name": "Punch List & Corrections", "wbs": "8.0",
             "start": 245, "duration": 15, "complete": 0, "cost": 80000, "critical": True, "crew": "General"},
            {"task_id": "8020", "task_name": "Final Inspection & Handover", "wbs": "8.0",
             "start": 255, "duration": 10, "complete": 0, "cost": 50000, "critical": True, "crew": "Management"},
        ]

        task_list = []
        for act in activities:
            start_date = base_date + timedelta(days=act["start"])
            end_date = start_date + timedelta(days=act["duration"])
            total_float = 0 if act["critical"] else (act["duration"] * 0.3)

            task_list.append({
                "task_id": act["task_id"],
                "task_name": act["task_name"],
                "wbs": act["wbs"],
                "start_date": start_date,
                "end_date": end_date,
                "original_duration": act["duration"],
                "remaining_duration": act["duration"] * (1 - act["complete"] / 100),
                "percent_complete": act["complete"],
                "budget_cost": act["cost"],
                "actual_cost": act["cost"] * (act["complete"] / 100),
                "total_float": total_float,
                "critical": act["critical"],
                "resource_crew": act["crew"],
                "status": "Complete" if act["complete"] == 100 else ("In Progress" if act["complete"] > 0 else "Not Started")
            })

        self.tasks = pd.DataFrame(task_list)

        relationships = []
        for i in range(1, len(activities)):
            pred_id = activities[i - 1]["task_id"]
            succ_id = activities[i]["task_id"]
            if i % 3 == 0 and i < len(activities) - 1:
                pred_id = activities[i - 2]["task_id"]
            relationships.append({
                "pred_task_id": pred_id,
                "succ_task_id": succ_id,
                "relationship_type": "FS",
                "lag_days": 0
            })
        self.relationships = pd.DataFrame(relationships)

        resources = []
        for _, task in self.tasks.iterrows():
            resources.append({
                "task_id": task["task_id"],
                "resource_id": f"RES_{task['resource_crew']}",
                "resource_name": task["resource_crew"] + " Crew",
                "units": 1.0,
                "cost_per_unit": task["budget_cost"] / max(1, task["original_duration"])
            })
        self.resources = pd.DataFrame(resources)

    def _parse_real_xer(self):
        """Parse a real XER file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        in_tasks = False
        in_relationships = False
        in_resources = False

        tasks_data = []
        relationships_data = []
        resources_data = []

        task_fields = []
        rel_fields = []
        res_fields = []

        for line in lines:
            line = line.strip()

            if line.startswith('%T'):
                if 'PROJECT' in line:
                    in_tasks = True
                    in_relationships = False
                    in_resources = False
                    task_fields = line.split('\t')[1:]
                elif 'TASKPRED' in line:
                    in_tasks = False
                    in_relationships = True
                    in_resources = False
                    rel_fields = line.split('\t')[1:]
                elif 'RES' in line:
                    in_tasks = False
                    in_relationships = False
                    in_resources = True
                    res_fields = line.split('\t')[1:]
                continue

            if not line or line.startswith('%'):
                in_tasks = False
                in_relationships = False
                in_resources = False
                continue

            values = line.split('\t')

            if in_tasks and len(values) >= len(task_fields):
                try:
                    task_dict = dict(zip(task_fields, values))
                    tasks_data.append({
                        "task_id": task_dict.get('task_id', ''),
                        "task_name": task_dict.get('task_name', ''),
                        "wbs": task_dict.get('wbs_id', ''),
                        "start_date": self._parse_xer_date(task_dict.get('task_start_date', '')),
                        "end_date": self._parse_xer_date(task_dict.get('task_finish_date', '')),
                        "original_duration": int(task_dict.get('task_duration', 0) or 0),
                        "remaining_duration": int(task_dict.get('task_remaining_drtn', 0) or 0),
                        "percent_complete": float(task_dict.get('task_pct_complete', 0) or 0),
                        "budget_cost": float(task_dict.get('task_cost', 0) or 0),
                        "actual_cost": float(task_dict.get('task_actual_cost', 0) or 0),
                        "total_float": float(task_dict.get('task_total_float', 0) or 0),
                        "critical": task_dict.get('task_critical', 'N') == 'Y',
                        "status": task_dict.get('task_status', 'NOT STARTED'),
                    })
                except (ValueError, IndexError):
                    continue

            elif in_relationships and len(values) >= len(rel_fields):
                try:
                    rel_dict = dict(zip(rel_fields, values))
                    relationships_data.append({
                        "pred_task_id": rel_dict.get('pred_task_id', ''),
                        "succ_task_id": rel_dict.get('succ_task_id', ''),
                        "relationship_type": rel_dict.get('pred_task_type', 'FS'),
                        "lag_days": int(rel_dict.get('lag_hr_cnt', 0) or 0) // 24
                    })
                except (ValueError, IndexError):
                    continue

            elif in_resources and len(values) >= len(res_fields):
                try:
                    res_dict = dict(zip(res_fields, values))
                    resources_data.append({
                        "task_id": res_dict.get('task_id', ''),
                        "resource_id": res_dict.get('resource_id', ''),
                        "resource_name": res_dict.get('resource_name', ''),
                        "units": float(res_dict.get('resource_units', 1) or 1),
                        "cost_per_unit": float(res_dict.get('resource_cost_per_unit', 0) or 0),
                    })
                except (ValueError, IndexError):
                    continue

        self.tasks = pd.DataFrame(tasks_data)
        self.relationships = pd.DataFrame(relationships_data)
        self.resources = pd.DataFrame(resources_data)

    def _parse_xer_date(self, date_str: str) -> datetime:
        """Parse XER date format."""
        if not date_str:
            return datetime.now()
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                return datetime.now()

    def get_schedule_summary(self) -> dict:
        """Get a summary of the schedule."""
        if self.tasks.empty:
            return {}

        total_tasks = len(self.tasks)
        completed = len(self.tasks[self.tasks['percent_complete'] == 100])
        in_progress = len(self.tasks[(self.tasks['percent_complete'] > 0) & (self.tasks['percent_complete'] < 100)])
        not_started = len(self.tasks[self.tasks['percent_complete'] == 0])
        critical_tasks = len(self.tasks[self.tasks['critical'] == True])

        total_budget = self.tasks['budget_cost'].sum()
        actual_spent = self.tasks['actual_cost'].sum()

        return {
            "total_tasks": total_tasks,
            "completed": completed,
            "in_progress": in_progress,
            "not_started": not_started,
            "critical_tasks": critical_tasks,
            "total_budget": total_budget,
            "actual_spent": actual_spent,
            "percent_complete": (actual_spent / total_budget * 100) if total_budget > 0 else 0,
            "start_date": self.tasks['start_date'].min(),
            "end_date": self.tasks['end_date'].max(),
        }

    def get_critical_path(self) -> list:
        """Get critical path tasks."""
        if self.tasks.empty:
            return []
        critical = self.tasks[self.tasks['critical'] == True].sort_values('start_date')
        return critical['task_id'].tolist()
