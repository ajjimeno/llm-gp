import sys

from deap import base, creator, gp

import gp_algorithm


def partial(func, *args):
    def wrapper():
        return func(*args)

    return wrapper


class Operation(object):
    pass


class COperation(Operation):
    pass


class Integer(object):
    pass


class WInteger(Integer):
    pass


class RInteger(Integer):
    pass


class SInteger(object):
    pass


class Status(object):
    pass


class Boolean(object):
    pass


def if_condition(comparison, out1, out2):
    out1() if comparison() else out2()


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)


def prog(out1):
    return partial(progn, out1)


class Runner:
    def __init__(self):
        self.__reset()

    def __reset(self):
        # print("reset")
        # self.memory = {}
        # self.memory_position = 0
        self.step = 0

        self.input_id = 0
        self.output_position_x = 0
        self.output_position_y = 0
        self.input_position_x = 0
        self.input_position_y = 0

        self.testing_output_position_x = 0
        self.testing_output_position_y = 0
        self.testing_input_position_x = 0
        self.testing_input_position_y = 0

        self.status = 0

        self.inner_loop = 0

    def _status_end(self):
        self.status = -1

    def status_end(self):
        return partial(self._status_end)

    def _set_status(self, value):
        self.status = value

    def set_status(self, value):
        return partial(self._set_status, value)

    def _check_status(self, value):
        return self.status == value

    def check_status(self, value):
        return partial(self._check_status, value)

    def _get_status(self):
        return self.status

    def get_status(self):
        return partial(self._get_status)

    def _get0(self):
        return 0

    def get0(self):
        return partial(self._get0)

    def _get1(self):
        return 1

    def get1(self):
        return partial(self._get1)

    def _get2(self):
        return 2

    def get2(self):
        return partial(self._get2)

    def _get3(self):
        return 3

    def get3(self):
        return partial(self._get3)

    def _get4(self):
        return 4

    def get4(self):
        return partial(self._get4)

    def _get5(self):
        return 5

    def get5(self):
        return partial(self._get5)

    def _get6(self):
        return 6

    def get6(self):
        return partial(self._get6)

    def _get7(self):
        return 7

    def get7(self):
        return partial(self._get7)

    def _get8(self):
        return 8

    def get8(self):
        return partial(self._get8)

    def _get9(self):
        return 9

    def get9(self):
        return partial(self._get9)

    def _get_1(self):
        return -1

    def get_1(self):
        return partial(self._get_1)

    def _loop(self, value, operation):
        v = value()
        self.inner_loop += 1

        if (
            self.inner_loop < 5 and v > 0 and v <= 30
        ):  # revise the limit with a better idea
            # if v > 0 and v <= 30: # revise the limit with a better idea
            for _ in range(v):
                operation()
        else:
            self.inner_loop += -1
            self.status = -2

        self.inner_loop += -1

    def loop(self, value, operation):
        return partial(self._loop, value, operation)

    def dowhile(self, value, operation):
        pass

    def _memory_inc(self):
        if self.memory_position not in self.memory:
            self.memory[self.memory_position] = 0

        self.memory[self.memory_position] += 1

    def memory_inc(self):
        return partial(self._memory_inc)

    def _memory_dec(self):
        if self.memory_position not in self.memory:
            self.memory[self.memory_position] = 0

        self.memory[self.memory_position] -= 1

    def memory_dec(self):
        return partial(self._memory_dec)

    def _read_memory(self):
        if self.memory_position not in self.memory:
            return 0
        else:
            return self.memory[self.memory_position]

    def read_memory(self):
        return partial(self._read_memory)

    def _write_memory(self, value):
        self.memory[self.memory_position] = value()

    def write_memory(self, value):
        return partial(self._write_memory, value)

    def _memory_move_left(self):
        self.memory_position -= 1

    def memory_move_left(self):
        return partial(self._memory_move_left)

    def _memory_move_right(self):
        self.memory_position += 1

    def memory_move_right(self):
        return partial(self._memory_move_right)

    def _input_end(self):
        return self.input_position_x == (
            len(self.input_training[self.input_id][0][self.input_position_y]) - 1
        )

    def input_end(self):
        return partial(self._input_end)

    def _input_down_end(self):
        return self.input_position_y == (len(self.input_training[self.input_id][0]) - 1)

    def input_max(self):
        pass

    def input_min(self):
        pass

    def input_beginning(self):
        pass

    def input_down_end(self):
        return partial(self._input_down_end)

    def _output_end(self):
        return self.output_position_x == (
            len(self.input_training[self.input_id][1][self.output_position_y]) - 1
        )

    def output_end(self):
        return partial(self._output_end)

    def output_beginning(self):
        pass

    def _output_down_end(self):
        return self.output_position_y == (
            len(self.input_training[self.input_id][1]) - 1
        )

    def output_down_end(self):
        return partial(self._output_down_end)

    def _output_move_left(self):
        if self.output_position_x > 0:
            self.output_position_x -= 1

    def output_move_left(self):
        return partial(self._output_move_left)

    def _output_move_right(self):
        if (
            self.output_position_x
            < len(self.input_training[self.input_id][1][self.output_position_y]) - 1
        ):
            self.output_position_x += 1

    def output_move_right(self):
        return partial(self._output_move_right)

    def _output_move_down(self):
        if self.output_position_y < len(self.input_training[self.input_id][1]) - 1:
            self.output_position_y += 1

    def output_move_down(self):
        return partial(self._output_move_down)

    def _output_move_up(self):
        if self.output_position_y > 0:
            self.output_position_y -= 1

    def output_move_up(self):
        return partial(self._output_move_up)

    def comparison(self, comparison, out1, out2):
        return partial(if_condition, comparison, out1, out2)

    def _bigger_than(self, input1, input2):
        return input1() > input2()

    def bigger_than(self, input1, input2):
        return partial(self._bigger_than, input1, input2)

    def bigger_thanW(self, input1, input2):
        pass

    def bigger_thanR(self, input1, input2):
        pass

    def _equal(self, input1, input2):
        return input1() > input2()

    def equal(self, input1, input2):
        return partial(self._equal, input1, input2)

    def equalW(self, input1, input2):
        pass

    def equalR(self, input1, input2):
        pass

    def no(self, input):
        pass

    def _get_input_position_x(self):
        return self.input_position_x

    def get_input_position_x(self):
        return partial(self._get_input_position_x)

    def _get_input_position_y(self):
        return self.input_position_y

    def get_input_position_y(self):
        return partial(self._get_input_position_y)

    def _get_output_position_x(self):
        return self.output_position_x

    def get_output_position_x(self):
        return partial(self._get_output_position_x)

    def _get_output_position_y(self):
        return self.output_position_y

    def get_output_position_y(self):
        return partial(self._get_output_position_y)

    def _get_length_input_x(self):
        return len(self.input_training[self.input_id][0][0])

    def get_length_input_x(self):
        return partial(self._get_length_input_x)

    def _get_length_input_y(self):
        return len(self.input_training[self.input_id][0])

    def get_length_input_y(self):
        return partial(self._get_length_input_y)

    def _get_length_output_x(self):
        return len(self.input_training[self.input_id][1][0])

    def get_length_output_x(self):
        return partial(self._get_length_output_x)

    def _get_length_output_y(self):
        return len(self.input_training[self.input_id][1])

    def get_length_output_y(self):
        return partial(self._get_length_output_y)

    # Input
    def _input_next(self):
        if self.input_id < len(self.input_training) - 1:
            self.input_id += 1
            self.input_position_x = 0
            self.input_position_y = 0
            self.output_position_x = 0
            self.output_position_y = 0

    def input_next(self):
        return partial(self._input_next)

    def _input_previous(self):
        if self.input_id > 0:
            self.input_id -= 1
            self.input_position_x = 0
            self.input_position_y = 0
            self.output_position_x = 0
            self.output_position_y = 0

    def input_previous(self):
        return partial(self._input_previous)

    def _input_move_left(self):
        if self.input_position_x > 0:
            self.input_position_x -= 1

    def input_move_left(self):
        return partial(self._input_move_left)

    def _input_move_right(self):
        if (
            self.input_position_x
            < len(self.input_training[self.input_id][0][self.input_position_y]) - 1
        ):
            self.input_position_x += 1

    def input_move_right(self):
        return partial(self._input_move_right)

    def _input_move_down(self):
        if self.input_position_y < len(self.input_training[self.input_id][0]) - 1:
            self.input_position_y += 1

    def input_move_down(self):
        return partial(self._input_move_down)

    def _input_move_up(self):
        if self.input_position_y > 0:
            self.input_position_y -= 1

    def input_move_up(self):
        return partial(self._input_move_up)

    def _input_read(self):
        try:
            test = self.input_training[self.input_id][0][self.input_position_y]
            return test[self.input_position_x]
        except:
            print(self.input_training[self.input_id][0])
            print(self.input_position_y, self.input_position_x)
            sys.exit(-1)

    def input_read(self):
        return partial(self._input_read)

    def _output_read(self):
        return self.input_training[self.input_id][1][self.output_position_y][
            self.output_position_x
        ]

    def output_read(self):
        return partial(self._output_read)

    def _reset_output_position(self):
        self.output_position_x = 0

    def reset_output_position(self):
        return partial(self._reset_output_position)

    def _reset_output_down_position(self):
        self.output_position_y = 0

    def reset_output_down_position(self):
        return partial(self._reset_output_down_position)

    def reset_input_position(self):
        pass

    def reset_input_down_position(self):
        pass

    def run(self, routine):
        self.__reset()

        # self.output = self.input.copy()
        while self.step < 900 and self.status >= 0:
            self.step += 1
            routine()

    # Testing
    def _get_testing_length_input_x(self):
        return len(self.input[0])

    def get_testing_length_input_x(self):
        return partial(self._get_length_input_x)

    def _get_testing_length_input_y(self):
        return len(self.input)

    def get_testing_length_input_y(self):
        return partial(self._get_length_input_y)

    def _get_testing_length_output_x(self):
        return len(self.output[0])

    def get_testing_length_output_x(self):
        return partial(self._get_length_output_x)

    def get_testing_length_output_y(self):
        return lambda: len(self.output)

    def _get_testing_input_position_y(self):
        return self.testing_output_position_y

    def get_testing_input_position_y(self):
        return partial(self._get_testing_input_position_y)

    def _get_testing_input_position_x(self):
        return self.testing_input_position_x

    def get_testing_input_position_x(self):
        return partial(self._get_testing_input_position_x)

    def _get_testing_output_position_y(self):
        return self.testing_output_position_y

    def get_testing_output_position_y(self):
        return partial(self._get_testing_output_position_y)

    def _get_testing_output_position_x(self):
        return self.testing_output_position_x

    def get_testing_output_position_x(self):
        return partial(self._get_testing_output_position_x)

    def _testing_input_read(self):
        return self.input[self.testing_input_position_y][self.testing_input_position_x]

    def testing_input_read(self):
        return partial(self._testing_input_read)

    def _testing_output_read_previous(self):
        return self.output[self.testing_output_position_y][
            self.testing_output_position_x - 1
        ]

    def testing_output_read_previous(self):
        return partial(self._testing_output_read_previous)

    def _testing_output_read(self):
        return self.output[self.testing_output_position_y][
            self.testing_output_position_x
        ]

    def testing_output_read(self):
        return partial(self._testing_output_read)

    def _testing_output_write_previous(self, value):
        self.output[
            self.testing_output_position_y, self.testing_output_position_x - 1
        ] = value()

    def testing_output_write_previous(self, value):
        return partial(self._testing_output_write_previous, value)

    def _testing_output_write(self, value):
        self.output[self.testing_output_position_y][
            self.testing_output_position_x
        ] = value()

    def testing_output_write(self, value):
        return partial(self._testing_output_write, value)

    def _testing_reset_output_position(self):
        self.testing_output_position_x = 0

    def testing_reset_output_position(self):
        return partial(self._testing_reset_output_position)

    def _testing_reset_output_down_position(self):
        self.testing_output_position_y = 0

    def testing_reset_output_down_position(self):
        return partial(self._testing_reset_output_down_position)

    def testing_reset_input_position(self):
        pass

    def testing_reset_input_down_position(self):
        pass

    def testing_input_max(self):
        pass

    def testing_input_min(self):
        pass

    def _testing_output_move_left(self):
        if self.testing_output_position_x > 0:
            self.testing_output_position_x -= 1

    def testing_output_move_left(self):
        return partial(self._testing_output_move_left)

    def _testing_output_move_right(self):
        if (
            self.testing_output_position_x
            < len(self.output[self.testing_output_position_y]) - 1
        ):
            self.testing_output_position_x += 1

    def testing_output_move_right(self):
        return partial(self._testing_output_move_right)

    def _testing_output_move_down(self):
        if self.testing_output_position_y < len(self.output) - 1:
            self.testing_output_position_y += 1

    def testing_output_move_down(self):
        return partial(self._testing_output_move_down)

    def _testing_output_move_up(self):
        if self.testing_output_position_y > 0:
            self.testing_output_position_y -= 1

    def testing_output_move_up(self):
        return partial(self._testing_output_move_up)

    def testing_is_output_end(self):
        return (
            lambda: self.testing_output_position_x
            == len(self.output[self.testing_output_position_y]) - 1
        )

    def _testing_is_output_down(self):
        return self.testing_output_position_y == len(self.output) - 1

    def testing_is_output_down(self):
        return partial(self._testing_is_output_down)

    def _testing_input_move_left(self):
        if self.testing_input_position_x > 0:
            self.testing_input_position_x -= 1

    def testing_input_move_left(self):
        return partial(self._testing_input_move_left)

    def _testing_input_move_right(self):
        if (
            self.testing_input_position_x
            < len(self.input[self.testing_input_position_y]) - 1
        ):
            self.testing_input_position_x += 1

    def testing_input_move_right(self):
        return partial(self._testing_input_move_right)

    def _testing_input_move_down(self):
        if self.testing_input_position_y < len(self.input) - 1:
            self.testing_input_position_y += 1

    def testing_input_move_down(self):
        return partial(self._testing_input_move_down)

    def _testing_input_move_up(self):
        if self.testing_input_position_y > 0:
            self.testing_input_position_y -= 1

    def testing_input_move_up(self):
        return partial(self._testing_input_move_up)

    # Check implementation in the future if required
    def bigger_than_output_next(self):
        pass

    def bigger_than_testing_output_next(self):
        pass

    def swap_testing_output_next(self):
        pass

    def testing_set_output_value(self, x, y, value):
        pass

    def testing_get_input_value(self, x, y):
        pass

    def testing_get_output_value(self, x, y):
        pass

    def testing_set_input_position(self, x, y):
        pass

    def testing_set_output_position(self, x, y):
        pass

    def in_input_shape(self):
        pass

    def aligned_above(self):
        pass

    def aligned_below(self):
        pass

    def aligned_left(self):
        pass

    def aligned_right(self):
        pass

    def add(self, x, y):
        pass

    def sub(self, x, y):
        pass

    def testing_output_distance_to_input_x(self):
        pass

    def testing_output_distance_to_input_y(self):
        pass

    def get_max_color(self):
        pass

    def stack_push(self, value):
        pass

    def stack_pop(self):
        pass

    def stack_top(self):
        pass


def set_pset(r, is_arc=True):
    pset = gp.PrimitiveSetTyped("MAIN", [], Operation)

    pset.addPrimitive(r.status_end, [], Operation)

    # pset.addPrimitive(r.check_status, [SInteger], Boolean)
    # pset.addPrimitive(r.set_status, [SInteger], Operation)
    # pset.addPrimitive(r.get_status, [], Status)

    pset.addPrimitive(r.get0, [], RInteger)
    pset.addPrimitive(r.get1, [], RInteger)
    pset.addPrimitive(r.get2, [], RInteger)
    pset.addPrimitive(r.get3, [], RInteger)
    pset.addPrimitive(r.get4, [], RInteger)
    pset.addPrimitive(r.get5, [], RInteger)
    pset.addPrimitive(r.get6, [], RInteger)
    pset.addPrimitive(r.get7, [], RInteger)
    pset.addPrimitive(r.get8, [], RInteger)
    pset.addPrimitive(r.get9, [], RInteger)

    """
    pset.addPrimitive(r.get_1, [], SInteger)
    """

    pset.addPrimitive(r.loop, [Integer, Operation], Operation)
    # pset.addPrimitive(r.dowhile, [Boolean, Operation], Operation)

    """
    pset.addPrimitive(r.memory_inc, [], Operation)
    pset.addPrimitive(r.memory_dec, [], Operation)
    pset.addPrimitive(r.read_memory, [RInteger], RInteger)
    pset.addPrimitive(r.write_memory, [RInteger, RInteger], Operation)
    pset.addPrimitive(r.memory_move_left, [], Operation)
    pset.addPrimitive(r.memory_move_right, [], Operation)
    """

    if not is_arc:
        pset.addPrimitive(r.input_next, [], Operation)
        pset.addPrimitive(r.input_previous, [], Operation)

        pset.addPrimitive(r.input_max, [], WInteger)
        pset.addPrimitive(r.input_min, [], WInteger)

        pset.addPrimitive(r.input_read, [], WInteger)
        pset.addPrimitive(r.output_read, [], WInteger)

        pset.addPrimitive(r.input_end, [], Boolean)
        pset.addPrimitive(r.input_beginning, [], Boolean)
        pset.addPrimitive(r.input_down_end, [], Boolean)
        pset.addPrimitive(r.input_move_left, [], Operation)
        pset.addPrimitive(r.input_move_right, [], Operation)

        # pset.addPrimitive(r.input_move_down, [], Operation)
        # pset.addPrimitive(r.input_move_up, [], Operation)
        # pset.addPrimitive(r.get_length_input_y, [], WInteger)
        # pset.addPrimitive(r.get_length_output_y, [], WInteger)

        """
        pset.addPrimitive(r.get_input_position_x, [], Integer)
        pset.addPrimitive(r.get_input_position_y, [], Integer)
        pset.addPrimitive(r.get_output_position_x, [], Integer)
        pset.addPrimitive(r.get_output_position_y, [], Integer)
        """

        pset.addPrimitive(r.get_length_input_x, [], WInteger)
        pset.addPrimitive(r.get_length_output_x, [], WInteger)

        pset.addPrimitive(r.output_end, [], Boolean)
        pset.addPrimitive(r.output_beginning, [], Boolean)
        # pset.addPrimitive(r.output_down_end, [], Boolean)
        pset.addPrimitive(r.output_move_left, [], Operation)
        pset.addPrimitive(r.output_move_right, [], Operation)
        # pset.addPrimitive(r.output_move_down, [], Operation)
        # pset.addPrimitive(r.output_move_up, [], Operation)
        pset.addPrimitive(r.reset_output_position, [], Operation)
        # pset.addPrimitive(r.reset_output_down_position, [], Operation)
        pset.addPrimitive(r.reset_input_position, [], Operation)
        # pset.addPrimitive(r.reset_input_down_position, [], Operation)
        pset.addPrimitive(r.bigger_thanW, [WInteger, WInteger], Boolean)
        pset.addPrimitive(r.equalW, [WInteger, WInteger], Boolean)
    else:
        pset.addPrimitive(
            r.testing_set_output_value,
            [RInteger, RInteger, RInteger],
            Operation,
        )
        pset.addPrimitive(r.testing_get_input_value, [RInteger, RInteger], RInteger)

        pset.addPrimitive(r.testing_get_output_value, [RInteger, RInteger], RInteger)

        pset.addPrimitive(r.testing_set_input_position, [RInteger, RInteger], Operation)
        pset.addPrimitive(
            r.testing_set_output_position, [RInteger, RInteger], Operation
        )

        pset.addPrimitive(r.in_input_shape, [], Boolean)

        pset.addPrimitive(r.aligned_above, [], RInteger)
        pset.addPrimitive(r.aligned_below, [], RInteger)
        pset.addPrimitive(r.aligned_left, [], RInteger)
        pset.addPrimitive(r.aligned_right, [], RInteger)

        pset.addPrimitive(r.add, [RInteger, RInteger], RInteger)
        pset.addPrimitive(r.sub, [RInteger, RInteger], RInteger)

        pset.addPrimitive(r.testing_output_distance_to_input_x, [], RInteger)
        pset.addPrimitive(r.testing_output_distance_to_input_y, [], RInteger)
        pset.addPrimitive(r.get_max_color, [], RInteger)

        pset.addPrimitive(r.testing_reset_output_down_position, [], Operation)
        pset.addPrimitive(r.testing_reset_input_down_position, [], Operation)
        pset.addPrimitive(r.testing_output_move_down, [], Operation)
        pset.addPrimitive(r.testing_output_move_up, [], Operation)
        pset.addPrimitive(r.get_testing_length_input_y, [], RInteger)
        pset.addPrimitive(r.get_testing_length_output_y, [], RInteger)
        pset.addPrimitive(r.get_testing_input_position_y, [], RInteger)
        pset.addPrimitive(r.get_testing_output_position_y, [], RInteger)
        pset.addPrimitive(r.testing_is_output_end, [], Boolean)
        pset.addPrimitive(r.testing_is_output_down, [], Boolean)
        pset.addPrimitive(r.testing_input_move_down, [], Operation)
        pset.addPrimitive(r.testing_input_move_up, [], Operation)

        pset.addPrimitive(r.stack_push, [RInteger], Operation)
        pset.addPrimitive(r.stack_pop, [], RInteger)
        pset.addPrimitive(r.stack_top, [], RInteger)

    pset.addPrimitive(prog2, [Operation, Operation], Operation)
    # pset.addPrimitive(prog3, [Operation, Operation, Operation], Operation)
    pset.addPrimitive(r.comparison, [Boolean, Operation, Operation], COperation)

    pset.addPrimitive(r.bigger_thanR, [RInteger, RInteger], Boolean)
    pset.addPrimitive(r.equalR, [RInteger, RInteger], Boolean)

    # pset.addPrimitive(r.no, [Boolean], Boolean)

    pset.addPrimitive(r.get_testing_length_input_x, [], RInteger)
    pset.addPrimitive(r.get_testing_length_output_x, [], RInteger)

    pset.addPrimitive(r.get_testing_input_position_x, [], RInteger)
    pset.addPrimitive(r.get_testing_output_position_x, [], RInteger)

    pset.addPrimitive(r.testing_input_read, [], RInteger)
    pset.addPrimitive(r.testing_output_read_previous, [], RInteger)
    pset.addPrimitive(r.testing_output_read, [], RInteger)
    pset.addPrimitive(r.testing_output_write_previous, [RInteger], Operation)
    pset.addPrimitive(r.testing_input_max, [], RInteger)
    pset.addPrimitive(r.testing_input_min, [], RInteger)
    pset.addPrimitive(r.testing_output_write, [RInteger], Operation)
    pset.addPrimitive(r.testing_reset_output_position, [], Operation)
    pset.addPrimitive(r.testing_reset_input_position, [], Operation)
    pset.addPrimitive(r.testing_output_move_left, [], Operation)
    pset.addPrimitive(r.testing_output_move_right, [], Operation)
    pset.addPrimitive(r.testing_input_move_left, [], Operation)
    pset.addPrimitive(r.testing_input_move_right, [], Operation)

    # Check implementation in the future if required
    pset.addPrimitive(r.bigger_than_output_next, [], Boolean)
    pset.addPrimitive(r.bigger_than_testing_output_next, [], Boolean)
    pset.addPrimitive(r.swap_testing_output_next, [], Operation)

    return pset


if __name__ == "__main__":

    r = Runner()
    pset = set_pset(r)

    creator.create("FitnessMin", base.Fitness, weights=(1.0, 0.000000000000005))

    creator.create("Individual", gp_algorithm.PrimitiveTree, fitness=creator.FitnessMin)

    a = creator.Individual.from_string("prog2(stack_push())", pset)
    print(a)

    print(pset.primitives.keys())
