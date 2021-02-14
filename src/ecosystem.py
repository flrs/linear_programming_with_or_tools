"""Ecosystem Class"""

from pathlib import Path
from pprint import pprint
from typing import Dict, Union

import pandas as pd
import plotly.express as px
from ortools.linear_solver import pywraplp
from plotly.graph_objs import Figure


class Ecosystem:
    def __init__(self,
                 market_def: Dict,
                 supply_def: Dict,
                 demand_def: Dict):
        self.market_penetration = None
        self.market_penetration_by_consumer = None
        self.market_size = None
        self.market_size_by_consumer = None
        self.market_captures_by_consumer = None

        self.supply_size = None
        self.supply_captures_by_supply_and_consumer = None
        self.supply_utilization_by_supply = None
        self.supply_utilization = None
        self.supply_utilization_by_consumer = None

        self.consumers = None

        self.market_def = market_def
        self.supply_def = supply_def
        self.demand_def = demand_def

        self.solver = pywraplp.Solver.CreateSolver('SCIP')

    @staticmethod
    def from_dict(dict_: Dict):
        try:
            return Ecosystem(
                market_def=dict_['market'],
                supply_def=dict_['supply'],
                demand_def=dict_['demand'])
        except KeyError:
            raise KeyError(f'If you create an Ecosystem from a dictionary, that dictionary needs to include the keys '
                           '"market", "demand" and "supply". The dictionary you supplied contains the following keys: '
                           '{}. Please fix the dictionary and try again.'.format(', '.join(list(dict.keys()))))

    @staticmethod
    def from_csv(path: Union[str, Path]):
        data = pd.read_csv(path, index_col=0)

        ecosystem_definition = {}
        ecosystem_definition['market'] = data.iloc[-1, :-1].astype(int).to_dict()
        ecosystem_definition['demand'] = data.iloc[:-1, :-1].to_dict()
        ecosystem_definition['supply'] = data.iloc[:-1, -1].astype(int).to_dict()

        return Ecosystem.from_dict(ecosystem_definition)

    def _test_suppliers_exist(self):
        for consumer_name, consumer_demand in self.demand_def.items():
            for supplier_name in consumer_demand.keys():
                assert supplier_name in self.supply_def, \
                    f'{supplier_name} is demanded by {consumer_name}, but it is not in supplier definition'

    def _test_supplier_geq_zero(self):
        for supplier_name, supplier_qty in self.supply_def.items():
            assert supplier_qty >= 0, f'Supplied quantity of {supplier_name} needs to be >= 0, but is {supplier_qty}'

    def _test_consumers_in_market(self):
        for consumer_name in self.demand_def.keys():
            assert consumer_name in self.market_def, f'{consumer_name} is in the demand definition, but not in the ' \
                                                     f'market definition.'

    def _test(self):
        self._test_suppliers_exist()
        self._test_supplier_geq_zero()
        self._test_consumers_in_market()

    def print_definition(self):
        print('-- Market --')
        pprint(self.market_def)
        print('-- Supply --')
        pprint(self.supply_def)
        print('-- Demand --')
        pprint(self.demand_def)

    def solve(self,
              print_solution: bool = True):
        self.consumers = {}
        for consumer_name, consumer_qty in self.market_def.items():
            self.consumers[consumer_name] = self.solver.IntVar(0, consumer_qty, consumer_name)

        def check_supplier_in_demands(supplier_name):
            supplier_in_demands = []
            for demands in self.demand_def.values():
                supplier_in_demands.append(supplier_name in demands)
            return any(supplier_in_demands)

        constraints = {}
        for supply_name, supply_qty in self.supply_def.items():
            if check_supplier_in_demands(supply_name):
                constraint = self.solver.RowConstraint(0, supply_qty, supply_name)
                constraints[supply_name] = constraint
                for consumer_name, consumer in self.consumers.items():
                    constraint.SetCoefficient(consumer, self.demand_def[consumer_name][supply_name])

        objective = self.solver.Objective()
        for consumer in self.consumers.values():
            objective.SetCoefficient(consumer, 1)
        objective.SetMaximization()

        status = self.solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError('The problem does not have an optimal solution.')

        self.market_size = sum(self.market_def.values())
        self.market_penetration = self.solver.Objective().Value() / self.market_size

        self.market_size_by_consumer = self.market_def
        self.market_captures_by_consumer = {consumer_name: consumer.solution_value() for consumer_name, consumer in
                                            self.consumers.items()}
        self.market_penetration_by_consumer = \
            {consumer_name: consumer.solution_value() / self.market_size_by_consumer[consumer_name]
             for consumer_name, consumer in self.consumers.items()}

        self.supply_size = sum(self.supply_def.values())
        self.supply_captures_by_supply_and_consumer = {}
        self.supply_utilization_by_supply = {}

        for constraint_name, constraint in constraints.items():
            captures_by_consumer = {consumer_name: constraint.GetCoefficient(consumer) * consumer.solution_value() for
                                    consumer_name, consumer in self.consumers.items()}
            self.supply_captures_by_supply_and_consumer[constraint_name] = captures_by_consumer
            self.supply_captures_by_supply_and_consumer[constraint_name]['unused'] = \
                self.supply_def[constraint_name] \
                - sum(self.supply_captures_by_supply_and_consumer[constraint_name].values())

            self.supply_utilization_by_supply[constraint_name] = \
                (sum(self.supply_captures_by_supply_and_consumer[constraint_name].values())
                 - self.supply_captures_by_supply_and_consumer[constraint_name]['unused']) \
                / sum(self.supply_captures_by_supply_and_consumer[constraint_name].values())
        self.supply_utilization = \
            1 - sum([supply_captures['unused']
                     for supply_name, supply_captures in self.supply_captures_by_supply_and_consumer.items()])\
            / sum([self.supply_def[supply_name] for supply_name in self.supply_captures_by_supply_and_consumer.keys()])

        self.supply_utilization_by_consumer = {}
        for consumer_name in self.consumers.keys():
            self.supply_utilization_by_consumer[consumer_name] = \
                sum([self.supply_captures_by_supply_and_consumer[supply_name][consumer_name]
                     for supply_name in self.supply_captures_by_supply_and_consumer.keys()]) \
                / self.supply_size
        self.supply_utilization_by_consumer['unused'] = 1 - sum(self.supply_utilization_by_consumer.values())

        if print_solution:
            self.print_solution()

    def print_solution(self):
        print('-- SOLUTION ––')

        print('Market penetration: {:.1%} ({:.0f}/{:.0f})'.format(
            self.market_penetration,
            self.solver.Objective().Value(),
            self.market_size))

        print('By consumer:')
        for consumer_name in sorted(self.market_penetration_by_consumer):
            print(' - {}: {:.1%} ({:.0f}/{:.0f})'.format(
                consumer_name.title(),
                self.market_penetration_by_consumer[consumer_name],
                self.consumers[consumer_name].solution_value(),
                self.market_size_by_consumer[consumer_name]
            ))
        print()
        print('Supply utilization: {:.1%} ({:.0f}/{:.0f})'.format(
            self.supply_utilization,
            self.supply_utilization * self.supply_size,
            self.supply_size
        ))
        print('By supply:')
        for supply_name, utilization in sorted(self.supply_utilization_by_supply.items()):
            print(' - {}: {:.1%} ({:.0f}/{:.0f})'.format(
                supply_name.title(),
                utilization,
                utilization * self.supply_def[supply_name],
                self.supply_def[supply_name]))
        print('By consumer:')
        for consumer_name, utilization in sorted(self.supply_utilization_by_consumer.items()):
            print(' - {}: {:.1%}'.format(
                consumer_name.title(),
                utilization))

    def plot_market_penetration(self) -> Figure:
        data = pd.Series(self.market_penetration_by_consumer).sort_values()
        data['overall'] = self.market_penetration
        data.index = [v.title() for v in data.index]
        fig = px.bar(data,
                     text=['{:.1%}'.format(v) for v in data.values],
                     orientation='h',
                     labels={'index': 'Consumer', 'value': 'Market Penetration'},
                     title='Market Penetration by Consumer')
        fig.update_layout(xaxis=dict(tickformat='%', range=[0, 1]), showlegend=False)
        fig.show()
        return fig

    def plot_supply_utilization(self, by: str = 'supply'):
        if by == 'supply':
            data = pd.Series(self.supply_utilization_by_supply).sort_values()
            data['overall'] = self.supply_utilization
            data.index = [v.title() for v in data.index]
            fig = px.bar(data,
                         text=['{:.1%}'.format(v) for v in data.values],
                         orientation='h',
                         labels={'index': 'Supply', 'value': 'Supply Utilization'},
                         title='Supply Utilization by Supply')
        elif by == 'consumer':
            data = pd.Series(self.supply_utilization_by_consumer).sort_values()
            data.index = [v.title() for v in data.index]
            fig = px.bar(data,
                         text=['{:.1%}'.format(v) for v in data.values],
                         orientation='h',
                         labels={'index': 'Consumer', 'value': 'Supply Utilization'},
                         title='Supply Utilization by Consumer')
            fig.update_layout(xaxis=dict(tickformat='%', range=[0, 1]), showlegend=False)
            fig.show()
            return fig
        else:
            raise KeyError('Argument `by` needs to be either "supply" or "consumer". '
                           'You provided value "{}"'.format(by))
        fig.update_layout(xaxis=dict(tickformat='%', range=[0, 1]), showlegend=False)
        fig.show()
        return fig
