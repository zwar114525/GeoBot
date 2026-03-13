from dataclasses import dataclass, field


@dataclass
class ReportSectionTemplate:
    number: str
    title: str
    required: bool = True
    subsections: list[dict] = field(default_factory=list)
    guidance: str = ""


STANDARD_REPORT_TEMPLATE = [
    ReportSectionTemplate(
        number="1",
        title="Introduction",
        subsections=[
            {"number": "1.1", "title": "Project Background"},
            {"number": "1.2", "title": "Scope of Work"},
            {"number": "1.3", "title": "Applicable Standards and Codes"},
        ],
        guidance="Describe project background, scope, and codes.",
    ),
    ReportSectionTemplate(
        number="2",
        title="Site Description",
        subsections=[
            {"number": "2.1", "title": "Site Location and Access"},
            {"number": "2.2", "title": "Topography and Geomorphology"},
            {"number": "2.3", "title": "Adjacent Structures and Services"},
        ],
        guidance="Describe location, topography, and nearby structures.",
    ),
    ReportSectionTemplate(
        number="3",
        title="Ground Investigation",
        subsections=[
            {"number": "3.1", "title": "Scope of Ground Investigation"},
            {"number": "3.2", "title": "Field Works"},
            {"number": "3.3", "title": "Laboratory Testing"},
        ],
        guidance="Summarise GI works and test data.",
    ),
    ReportSectionTemplate(
        number="4",
        title="Geotechnical Conditions",
        subsections=[
            {"number": "4.1", "title": "Stratigraphy and Ground Model"},
            {"number": "4.2", "title": "Soil and Rock Parameters"},
            {"number": "4.3", "title": "Groundwater Conditions"},
            {"number": "4.4", "title": "Geological Hazards"},
        ],
        guidance="Describe ground model and parameters.",
    ),
    ReportSectionTemplate(
        number="5",
        title="Analysis and Design",
        subsections=[
            {"number": "5.1", "title": "Design Basis and Load Cases"},
            {"number": "5.2", "title": "Foundation Design"},
            {"number": "5.3", "title": "Slope Stability Assessment"},
            {"number": "5.4", "title": "Retaining Wall Design"},
            {"number": "5.5", "title": "Settlement Assessment"},
            {"number": "5.6", "title": "Seismic Considerations"},
        ],
        guidance="Present methods, inputs, results, and checks.",
    ),
    ReportSectionTemplate(
        number="6",
        title="Recommendations",
        subsections=[
            {"number": "6.1", "title": "Foundation Recommendations"},
            {"number": "6.2", "title": "Earthworks Recommendations"},
            {"number": "6.3", "title": "Slope Protection and Drainage"},
            {"number": "6.4", "title": "Geotechnical Constraints"},
        ],
        guidance="Provide clear actionable recommendations.",
    ),
    ReportSectionTemplate(
        number="7",
        title="Construction Considerations",
        subsections=[
            {"number": "7.1", "title": "Excavation and Temporary Support"},
            {"number": "7.2", "title": "Construction Sequence"},
            {"number": "7.3", "title": "Monitoring Requirements"},
        ],
        guidance="Describe construction risks and controls.",
    ),
    ReportSectionTemplate(number="8", title="References", guidance="List references."),
    ReportSectionTemplate(number="Appendix A", title="Borehole Logs and Test Results"),
    ReportSectionTemplate(number="Appendix B", title="Design Calculations"),
    ReportSectionTemplate(number="Appendix C", title="Drawings", required=False),
]
