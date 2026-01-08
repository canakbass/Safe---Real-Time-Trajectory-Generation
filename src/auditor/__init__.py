"""
Auditor Module
==============
Contains the EnergyAuditor for benchmarking trajectory solvers.
"""

from .energy_auditor import (
    EnergyAuditor,
    AuditReport,
    PowerProfile,
)

__all__ = [
    'EnergyAuditor',
    'AuditReport',
    'PowerProfile',
]
