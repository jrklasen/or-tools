// Copyright 2010-2025 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ortools/constraint_solver/constraint_solver.h"

#include <setjmp.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "ortools/constraint_solver/assignment.pb.h"
#include "ortools/constraint_solver/python/constraint_solver_doc.h"
#include "pybind11/cast.h"
#include "pybind11/functional.h"
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace py = ::pybind11;

using ::operations_research::Assignment;
using ::operations_research::AssignmentProto;
using ::operations_research::BaseObject;
using ::operations_research::Constraint;
using ::operations_research::ConstraintSolverParameters;
using ::operations_research::DecisionBuilder;
using ::operations_research::IntervalVar;
using ::operations_research::IntExpr;
using ::operations_research::IntVar;
using ::operations_research::ModelVisitor;
using ::operations_research::PropagationBaseObject;
using ::operations_research::Solver;

// There is no proper error propagation in `constraint_solver` but some
// operation may fail and end-up calling `Solver::Fail()`. `Solver` offers a
// `set_fail_intercept` method we can use to _break_ and return control flow to
// the caller in this case. Theoretically, we could just `set_fail_intercept` to
// `throw` but `Solver` is not compiled with exception enabled, this would
// result in a crash. Instead we use `setjmp`/`longjump` to resume the control
// flow in the function below and throw from here. This is quite convoluted but
// a cleaner solution would require rewriting the API.
template <typename T>
void ThrowOnFailure(T* this_, absl::FunctionRef<void(T*)> action) {
  Solver* solver = this_->solver();
  jmp_buf buffer;
  solver->set_fail_intercept([&buffer]() { longjmp(buffer, 1); });
  auto cleanup =
      absl::MakeCleanup([solver]() { solver->clear_fail_intercept(); });
  if (setjmp(buffer) == 0) {
    // If the statement below end-up calling `Solver::Fail()`, `longjmp` is
    // executed and resets execution to the if-condition above. The code then
    // branches in the `else` clause below and throws.
    action(this_);
  } else {
    throw py::value_error("Solver fails outside of solve()");
  }
}

class BaseObjectPythonHelper {
 public:
  static std::string DebugString(BaseObject* this_) {
    return this_->DebugString();
  }
};

class PropagationBaseObjectPythonHelper : BaseObjectPythonHelper {
 public:
  static std::string DebugString(PropagationBaseObject* this_) {
    return this_->DebugString();
  }

  static Solver* solver(PropagationBaseObject* this_) {
    return this_->solver();
  }

  static std::string name(PropagationBaseObject* this_) {
    return this_->name();
  }

  static void SetName(PropagationBaseObject* this_, absl::string_view name) {
    this_->set_name(name);
  }
};

class IntExprPythonHelper : PropagationBaseObjectPythonHelper {
 public:
  static int64_t Min(IntExpr* this_) { return this_->Min(); }

  static int64_t Max(IntExpr* this_) { return this_->Max(); }

  static void SetMin(IntExpr* this_, int64_t m) {
    ThrowOnFailure<IntExpr>(this_, [m](IntExpr* this_) { this_->SetMin(m); });
  }

  static void SetMax(IntExpr* this_, int64_t m) {
    ThrowOnFailure<IntExpr>(this_, [m](IntExpr* this_) { this_->SetMax(m); });
  }

  static void SetRange(IntExpr* this_, int64_t mi, int64_t ma) {
    ThrowOnFailure<IntExpr>(
        this_, [mi, ma](IntExpr* this_) { this_->SetRange(mi, ma); });
  }

  static void SetValue(IntExpr* this_, int64_t v) {
    ThrowOnFailure<IntExpr>(this_, [v](IntExpr* this_) { this_->SetValue(v); });
  }

  static bool Bound(IntExpr* this_) { return this_->Bound(); }
};

class IntVarPythonHelper : IntExprPythonHelper {
 public:
  static std::string name(IntVar* this_) { return this_->name(); }

  static int64_t Value(IntVar* this_) { return this_->Value(); }

  static void RemoveValue(IntVar* this_, int64_t v) {
    ThrowOnFailure<IntVar>(this_,
                           [v](IntVar* this_) { this_->RemoveValue(v); });
  }

  static int64_t Size(IntVar* this_) { return this_->Size(); }
};

PYBIND11_MODULE(constraint_solver, m) {
  pybind11_protobuf::ImportNativeProtoCasters();

  py::enum_<Solver::IntVarStrategy>(m, "IntVarStrategy")
      .value("INT_VAR_DEFAULT", Solver::INT_VAR_DEFAULT)
      .value("INT_VAR_SIMPLE", Solver::INT_VAR_SIMPLE)
      .value("CHOOSE_FIRST_UNBOUND", Solver::CHOOSE_FIRST_UNBOUND)
      .value("CHOOSE_RANDOM", Solver::CHOOSE_RANDOM)
      .value("CHOOSE_MIN_SIZE_LOWEST_MIN", Solver::CHOOSE_MIN_SIZE_LOWEST_MIN)
      .value("CHOOSE_MIN_SIZE_HIGHEST_MIN", Solver::CHOOSE_MIN_SIZE_HIGHEST_MIN)
      .value("CHOOSE_MIN_SIZE_LOWEST_MAX", Solver::CHOOSE_MIN_SIZE_LOWEST_MAX)
      .value("CHOOSE_MIN_SIZE_HIGHEST_MAX", Solver::CHOOSE_MIN_SIZE_HIGHEST_MAX)
      .value("CHOOSE_LOWEST_MIN", Solver::CHOOSE_LOWEST_MIN)
      .value("CHOOSE_HIGHEST_MAX", Solver::CHOOSE_HIGHEST_MAX)
      .value("CHOOSE_MIN_SIZE", Solver::CHOOSE_MIN_SIZE)
      .value("CHOOSE_MAX_SIZE", Solver::CHOOSE_MAX_SIZE)
      .value("CHOOSE_MAX_REGRET_ON_MIN", Solver::CHOOSE_MAX_REGRET_ON_MIN)
      .value("CHOOSE_PATH", Solver::CHOOSE_PATH)
      .export_values();

  py::enum_<Solver::IntValueStrategy>(m, "IntValueStrategy")
      .value("INT_VALUE_DEFAULT", Solver::INT_VALUE_DEFAULT)
      .value("INT_VALUE_SIMPLE", Solver::INT_VALUE_SIMPLE)
      .value("ASSIGN_MIN_VALUE", Solver::ASSIGN_MIN_VALUE)
      .value("ASSIGN_MAX_VALUE", Solver::ASSIGN_MAX_VALUE)
      .value("ASSIGN_RANDOM_VALUE", Solver::ASSIGN_RANDOM_VALUE)
      .value("ASSIGN_CENTER_VALUE", Solver::ASSIGN_CENTER_VALUE)
      .value("SPLIT_LOWER_HALF", Solver::SPLIT_LOWER_HALF)
      .value("SPLIT_UPPER_HALF", Solver::SPLIT_UPPER_HALF)
      .export_values();

  pybind11::enum_<Solver::UnaryIntervalRelation>(m, "UnaryIntervalRelation")
      .value("ENDS_AFTER", Solver::ENDS_AFTER)
      .value("ENDS_AT", Solver::ENDS_AT)
      .value("ENDS_BEFORE", Solver::ENDS_BEFORE)
      .value("STARTS_AFTER", Solver::STARTS_AFTER)
      .value("STARTS_AT", Solver::STARTS_AT)
      .value("STARTS_BEFORE", Solver::STARTS_BEFORE)
      .value("CROSS_DATE", Solver::CROSS_DATE)
      .value("AVOID_DATE", Solver::AVOID_DATE)
      .export_values();

  pybind11::enum_<Solver::BinaryIntervalRelation>(m, "BinaryIntervalRelation")
      .value("ENDS_AFTER_END", Solver::ENDS_AFTER_END)
      .value("ENDS_AFTER_START", Solver::ENDS_AFTER_START)
      .value("ENDS_AT_END", Solver::ENDS_AT_END)
      .value("ENDS_AT_START", Solver::ENDS_AT_START)
      .value("STARTS_AFTER_END", Solver::STARTS_AFTER_END)
      .value("STARTS_AFTER_START", Solver::STARTS_AFTER_START)
      .value("STARTS_AT_END", Solver::STARTS_AT_END)
      .value("STARTS_AT_START", Solver::STARTS_AT_START)
      .value("STAYS_IN_SYNC", Solver::STAYS_IN_SYNC)
      .export_values();

  py::class_<Solver>(m, "Solver", DOC(operations_research, Solver))
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&,
                          const ConstraintSolverParameters&>())
      .def("__str__", &Solver::DebugString)
      .def("default_solver_parameters", &Solver::DefaultSolverParameters)
      .def("parameters", &Solver::parameters)
      .def("local_search_profile", &Solver::LocalSearchProfile)
      .def("new_int_var",
           py::overload_cast<int64_t, int64_t, const std::string&>(
               &Solver::MakeIntVar),
           DOC(operations_research, Solver, MakeIntVar),
           py::return_value_policy::reference_internal)
      .def("new_int_var",
           py::overload_cast<int64_t, int64_t>(&Solver::MakeIntVar),
           DOC(operations_research, Solver, MakeIntVar),
           py::return_value_policy::reference_internal)
      .def("new_int_var",
           py::overload_cast<const std::vector<int64_t>&,
                                   const std::string&>(&Solver::MakeIntVar),
           DOC(operations_research, Solver, MakeIntVar_2),
           py::return_value_policy::reference_internal)
      .def("new_int_var",
           py::overload_cast<const std::vector<int64_t>&>(
               &Solver::MakeIntVar),
           DOC(operations_research, Solver, MakeIntVar_2),
           py::return_value_policy::reference_internal)
      .def("new_interval_var",
           py::overload_cast<int64_t, int64_t, int64_t, int64_t, int64_t,
                                   int64_t, bool, const std::string&>(
               &Solver::MakeIntervalVar),
           py::arg("start_min"),
           py::arg("start_max"),
           py::arg("duration_min"),
           py::arg("duration_max"),
           py::arg("end_min"),
           py::arg("end_max"),
           py::arg("optional"),
           py::arg("name"),
           DOC(operations_research, Solver, MakeIntervalVar),
           py::return_value_policy::reference_internal)
      .def("new_fixed_duration_interval_var",
           py::overload_cast<int64_t, int64_t, int64_t, bool,
                                   const std::string&>(
               &Solver::MakeFixedDurationIntervalVar),
           DOC(operations_research, Solver, MakeFixedDurationIntervalVar),
           py::return_value_policy::reference_internal)
      .def("new_fixed_duration_interval_var",
           py::overload_cast<IntVar*, int64_t, const std::string&>(
               &Solver::MakeFixedDurationIntervalVar),
           DOC(operations_research, Solver, MakeFixedDurationIntervalVar_2),
           py::return_value_policy::reference_internal)
      .def("new_fixed_duration_interval_var",
           py::overload_cast<IntVar*, int64_t, IntVar*,
                                   const std::string&>(
               &Solver::MakeFixedDurationIntervalVar),
           DOC(operations_research, Solver, MakeFixedDurationIntervalVar_3),
           py::return_value_policy::reference_internal)
      .def("add", &Solver::AddConstraint,
           DOC(operations_research, Solver, AddConstraint), py::arg("c"))
      .def("add_abs_equality",
           [](Solver* s, IntVar* var, IntVar* abs_var) {
             s->AddConstraint(s->MakeAbsEquality(var, abs_var));
           })
      .def("add_all_different",
           [](Solver* s, const std::vector<IntVar*>& vars) {
             s->AddConstraint(s->MakeAllDifferent(vars));
           })
      .def("add_all_different",
           [](Solver* s, const std::vector<IntVar*>& vars, bool stronger) {
             s->AddConstraint(s->MakeAllDifferent(vars, stronger));
           })
      .def("add_all_different_except",
           [](Solver* s, const std::vector<IntVar*>& vars, int64_t v) {
             s->AddConstraint(s->MakeAllDifferentExcept(vars, v));
           })
      .def("add_between_ct",
           [](Solver* s, IntExpr* expr, int64_t l, int64_t u) {
             s->AddConstraint(s->MakeBetweenCt(expr, l, u));
           })
      .def("add_circuit",
           [](Solver* s, const std::vector<IntVar*>& nexts) {
             s->AddConstraint(s->MakeCircuit(nexts));
           })
      .def("add_count",
           [](Solver* s, const std::vector<IntVar*>& vars, int64_t value,
              int64_t max_count) {
             s->AddConstraint(s->MakeCount(vars, value, max_count));
           })
      .def("add_count",
           [](Solver* s, const std::vector<IntVar*>& vars, int64_t value,
              IntVar* max_count) {
             s->AddConstraint(s->MakeCount(vars, value, max_count));
           })
      .def("add_cover",
           [](Solver* s, const std::vector<IntervalVar*>& vars,
              IntervalVar* target_var) {
             s->AddConstraint(s->MakeCover(vars, target_var));
           })
      .def("add_cumulative",
           [](Solver* s, const std::vector<IntervalVar*>& intervals,
              const std::vector<int64_t>& demands, int64_t capacity,
              const std::string& name) {
             s->AddConstraint(
                 s->MakeCumulative(intervals, demands, capacity, name));
           })
      .def("add_cumulative",
           [](Solver* s, const std::vector<IntervalVar*>& intervals,
              const std::vector<int>& demands, int64_t capacity,
              const std::string& name) {
             s->AddConstraint(
                 s->MakeCumulative(intervals, demands, capacity, name));
           })
      .def("add_cumulative",
           [](Solver* s, const std::vector<IntervalVar*>& intervals,
              const std::vector<int64_t>& demands, IntVar* capacity,
              absl::string_view name) {
             s->AddConstraint(
                 s->MakeCumulative(intervals, demands, capacity, name));
           })
      .def("add_cumulative",
           [](Solver* s, const std::vector<IntervalVar*>& intervals,
              const std::vector<int>& demands, IntVar* capacity,
              const std::string& name) {
             s->AddConstraint(
                 s->MakeCumulative(intervals, demands, capacity, name));
           })
      .def("add_cumulative",
           [](Solver* s, const std::vector<IntervalVar*>& intervals,
              const std::vector<IntVar*>& demands, int64_t capacity,
              const std::string& name) {
             s->AddConstraint(
                 s->MakeCumulative(intervals, demands, capacity, name));
           })
      .def("add_cumulative",
           [](Solver* s, const std::vector<IntervalVar*>& intervals,
              const std::vector<IntVar*>& demands, IntVar* capacity,
              const std::string& name) {
             s->AddConstraint(
                 s->MakeCumulative(intervals, demands, capacity, name));
           })
      .def("add_delayed_path_cumul",
           [](Solver* s, const std::vector<IntVar*>& nexts,
              const std::vector<IntVar*>& active,
              const std::vector<IntVar*>& cumuls,
              const std::vector<IntVar*>& transits) {
             s->AddConstraint(s->MakeDelayedPathCumul(nexts, active, cumuls,
                                                      transits));
           })
      .def("add_deviation",
           [](Solver* s, const std::vector<IntVar*>& vars,
              IntVar* deviation_var, int64_t total_sum) {
             s->AddConstraint(
                 s->MakeDeviation(vars, deviation_var, total_sum));
           })
      .def("add_disjunctive_constraint",
           [](Solver* s, const std::vector<IntervalVar*>& intervals,
              const std::string& name) {
             s->AddConstraint(s->MakeDisjunctiveConstraint(intervals, name));
           })
      .def("add_distribute",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int64_t>& values,
              const std::vector<IntVar*>& cards) {
             s->AddConstraint(s->MakeDistribute(vars, values, cards));
           })
      .def("add_distribute",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int>& values,
              const std::vector<IntVar*>& cards) {
             s->AddConstraint(s->MakeDistribute(vars, values, cards));
           })
      .def("add_distribute",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<IntVar*>& cards) {
             s->AddConstraint(s->MakeDistribute(vars, cards));
           })
      .def("add_distribute",
           [](Solver* s, const std::vector<IntVar*>& vars, int64_t card_min,
              int64_t card_max, int64_t card_size) {
             s->AddConstraint(
                 s->MakeDistribute(vars, card_min, card_max, card_size));
           })
      .def("add_distribute",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int64_t>& values,
              const std::vector<int64_t>& cards) {
             s->AddConstraint(s->MakeDistribute(vars, values, cards));
           })
      .def("add_distribute",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int>& values,
              const std::vector<int>& cards) {
             s->AddConstraint(s->MakeDistribute(vars, values, cards));
           })
      .def("add_element_equality",
           [](Solver* s, const std::vector<int64_t>& vals, IntVar* index,
              IntVar* target) {
             s->AddConstraint(s->MakeElementEquality(vals, index, target));
           })
      .def("add_element_equality",
           [](Solver* s, const std::vector<int>& vals, IntVar* index,
              IntVar* target) {
             s->AddConstraint(s->MakeElementEquality(vals, index, target));
           })
      .def("add_element_equality",
           [](Solver* s, const std::vector<IntVar*>& vars, IntVar* index,
              IntVar* target) {
             s->AddConstraint(s->MakeElementEquality(vars, index, target));
           })
      .def("add_element_equality",
           [](Solver* s, const std::vector<IntVar*>& vars, IntVar* index,
              int64_t target) {
             s->AddConstraint(s->MakeElementEquality(vars, index, target));
           })
      .def("add_false_constraint",
           [](Solver* s) { s->AddConstraint(s->MakeFalseConstraint()); })
      .def("add_index_of_constraint",
           [](Solver* s, const std::vector<IntVar*>& vars, IntVar* index,
              int64_t target) {
             s->AddConstraint(s->MakeIndexOfConstraint(vars, index, target));
           })
      .def("add_interval_var_relation",
           [](Solver* s, IntervalVar* t, Solver::UnaryIntervalRelation r,
              int64_t d) {
             s->AddConstraint(s->MakeIntervalVarRelation(t, r, d));
           })
      .def("add_interval_var_relation",
           [](Solver* s, IntervalVar* t1, Solver::BinaryIntervalRelation r,
              IntervalVar* t2) {
             s->AddConstraint(s->MakeIntervalVarRelation(t1, r, t2));
           })
      .def("add_interval_var_relation",
           [](Solver* s, IntervalVar* t1, Solver::BinaryIntervalRelation r,
              IntervalVar* t2, int64_t delay) {
             s->AddConstraint(s->MakeIntervalVarRelationWithDelay(t1, r, t2,
                                                                  delay));
           })
      .def("add_inverse_permutation_constraint",
           [](Solver* s, const std::vector<IntVar*>& left,
              const std::vector<IntVar*>& right) {
             s->AddConstraint(s->MakeInversePermutationConstraint(left, right));
           })
      .def("add_is_between_ct",
           [](Solver* s, IntExpr* var, int64_t l, int64_t u, IntVar* boolvar) {
             s->AddConstraint(s->MakeIsBetweenCt(var, l, u, boolvar));
           })
      .def("add_is_different_cst_ct",
           [](Solver* s, IntExpr* var, int64_t value, IntVar* boolvar) {
             s->AddConstraint(s->MakeIsDifferentCstCt(var, value, boolvar));
           })
      .def("add_is_different_ct",
           [](Solver* s, IntExpr* var, IntExpr* other, IntVar* boolvar) {
             s->AddConstraint(s->MakeIsDifferentCt(var, other, boolvar));
           })
      .def("add_is_equal_cst_ct",
           [](Solver* s, IntExpr* var, int64_t value, IntVar* boolvar) {
             s->AddConstraint(s->MakeIsEqualCstCt(var, value, boolvar));
           })
      .def("add_is_equal_ct",
           [](Solver* s, IntExpr* var, IntExpr* other, IntVar* boolvar) {
             s->AddConstraint(s->MakeIsEqualCt(var, other, boolvar));
           })
      .def("add_is_greater_cst_ct",
           [](Solver* s, IntExpr* var, int64_t value, IntVar* boolvar) {
             s->AddConstraint(s->MakeIsGreaterCstCt(var, value, boolvar));
           })
      .def("add_is_greater_ct",
           [](Solver* s, IntExpr* var, IntExpr* other, IntVar* boolvar) {
             s->AddConstraint(s->MakeIsGreaterCt(var, other, boolvar));
           })
      .def("add_is_greater_or_equal_cst_ct",
           [](Solver* s, IntExpr* var, int64_t value, IntVar* boolvar) {
             s->AddConstraint(
                 s->MakeIsGreaterOrEqualCstCt(var, value, boolvar));
           })
      .def("add_is_greater_or_equal_ct",
           [](Solver* s, IntExpr* var, IntExpr* other, IntVar* boolvar) {
             s->AddConstraint(
                 s->MakeIsGreaterOrEqualCt(var, other, boolvar));
           })
      .def("add_is_less_cst_ct",
           [](Solver* s, IntExpr* var, int64_t value, IntVar* boolvar) {
             s->AddConstraint(s->MakeIsLessCstCt(var, value, boolvar));
           })
      .def("add_is_less_ct",
           [](Solver* s, IntExpr* var, IntExpr* other, IntVar* boolvar) {
             s->AddConstraint(s->MakeIsLessCt(var, other, boolvar));
           })
      .def("add_is_less_or_equal_cst_ct",
           [](Solver* s, IntExpr* var, int64_t value, IntVar* boolvar) {
             s->AddConstraint(
                 s->MakeIsLessOrEqualCstCt(var, value, boolvar));
           })
      .def("add_is_less_or_equal_ct",
           [](Solver* s, IntExpr* var, IntExpr* other, IntVar* boolvar) {
             s->AddConstraint(s->MakeIsLessOrEqualCt(var, other, boolvar));
           })
      .def("add_is_member_ct",
           [](Solver* s, IntExpr* var, const std::vector<int64_t>& values,
              IntVar* boolvar) {
             s->AddConstraint(s->MakeIsMemberCt(var, values, boolvar));
           })
      .def("add_lexical_less",
           [](Solver* s, const std::vector<IntVar*>& left,
              const std::vector<IntVar*>& right) {
             s->AddConstraint(s->MakeLexicalLess(left, right));
           })
      .def("add_lexical_less_or_equal",
           [](Solver* s, const std::vector<IntVar*>& left,
              const std::vector<IntVar*>& right) {
             s->AddConstraint(s->MakeLexicalLessOrEqual(left, right));
           })
      .def("add_max_equality",
           [](Solver* s, const std::vector<IntVar*>& vars, IntVar* var) {
             s->AddConstraint(s->MakeMaxEquality(vars, var));
           })
      .def("add_member_ct",
           [](Solver* s, IntExpr* expr, const std::vector<int64_t>& values) {
             s->AddConstraint(s->MakeMemberCt(expr, values));
           })
      .def("add_min_equality",
           [](Solver* s, const std::vector<IntVar*>& vars, IntVar* var) {
             s->AddConstraint(s->MakeMinEquality(vars, var));
           })
      .def("add_non_overlapping_boxes_constraint",
           [](Solver* s, const std::vector<IntVar*>& x_vars,
              const std::vector<IntVar*>& y_vars,
              const std::vector<IntVar*>& x_size,
              const std::vector<IntVar*>& y_size) {
             s->AddConstraint(s->MakeNonOverlappingBoxesConstraint(
                 x_vars, y_vars, x_size, y_size));
           })
      .def("add_non_overlapping_boxes_constraint",
           [](Solver* s, const std::vector<IntVar*>& x_vars,
              const std::vector<IntVar*>& y_vars,
              const std::vector<int64_t>& x_size,
              const std::vector<int64_t>& y_size) {
             s->AddConstraint(s->MakeNonOverlappingBoxesConstraint(
                 x_vars, y_vars, x_size, y_size));
           })
      .def("add_non_overlapping_boxes_constraint",
           [](Solver* s, const std::vector<IntVar*>& x_vars,
              const std::vector<IntVar*>& y_vars,
              const std::vector<int>& x_size, const std::vector<int>& y_size) {
             s->AddConstraint(s->MakeNonOverlappingBoxesConstraint(
                 x_vars, y_vars, x_size, y_size));
           })
      .def("add_not_member_ct",
           [](Solver* s, IntExpr* expr, const std::vector<int64_t>& values) {
             s->AddConstraint(s->MakeNotMemberCt(expr, values));
           })
      .def("add_null_intersect",
           [](Solver* s, const std::vector<IntVar*>& first_vars,
              const std::vector<IntVar*>& second_vars) {
             s->AddConstraint(
                 s->MakeNullIntersect(first_vars, second_vars));
           })
      .def("add_null_intersect_except",
           [](Solver* s, const std::vector<IntVar*>& first_vars,
              const std::vector<IntVar*>& second_vars, int64_t escape_value) {
             s->AddConstraint(s->MakeNullIntersectExcept(
                 first_vars, second_vars, escape_value));
           })
      .def("add_pack",
           [](Solver* s, const std::vector<IntVar*>& vars, int number_of_bins) {
             s->AddConstraint(s->MakePack(vars, number_of_bins));
           })
      .def("add_path_cumul",
           [](Solver* s, const std::vector<IntVar*>& nexts,
              const std::vector<IntVar*>& active,
              const std::vector<IntVar*>& cumuls,
              const std::vector<IntVar*>& transits) {
             s->AddConstraint(
                 s->MakePathCumul(nexts, active, cumuls, transits));
           })
      .def("add_path_cumul",
           [](Solver* s, const std::vector<IntVar*>& nexts,
              const std::vector<IntVar*>& active,
              const std::vector<IntVar*>& cumuls,
              Solver::IndexEvaluator2 transit_evaluator) {
             s->AddConstraint(
                 s->MakePathCumul(nexts, active, cumuls, transit_evaluator));
           })
      .def("add_scal_prod_equality",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int64_t>& coeffs, int64_t cst) {
             s->AddConstraint(s->MakeScalProdEquality(vars, coeffs, cst));
           })
      .def("add_scal_prod_equality",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int>& coeffs, int64_t cst) {
             s->AddConstraint(s->MakeScalProdEquality(vars, coeffs, cst));
           })
      .def("add_scal_prod_equality",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int64_t>& coeffs, IntVar* target) {
             s->AddConstraint(s->MakeScalProdEquality(vars, coeffs, target));
           })
      .def("add_scal_prod_equality",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int>& coeffs, IntVar* target) {
             s->AddConstraint(s->MakeScalProdEquality(vars, coeffs, target));
           })
      .def("add_scal_prod_greater_or_equal",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int64_t>& coeffs, int64_t cst) {
             s->AddConstraint(
                 s->MakeScalProdGreaterOrEqual(vars, coeffs, cst));
           })
      .def("add_scal_prod_greater_or_equal",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int>& coeffs, int64_t cst) {
             s->AddConstraint(
                 s->MakeScalProdGreaterOrEqual(vars, coeffs, cst));
           })
      .def("add_scal_prod_less_or_equal",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int64_t>& coeffs, int64_t cst) {
             s->AddConstraint(
                 s->MakeScalProdLessOrEqual(vars, coeffs, cst));
           })
      .def("add_scal_prod_less_or_equal",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<int>& coeffs, int64_t cst) {
             s->AddConstraint(
                 s->MakeScalProdLessOrEqual(vars, coeffs, cst));
           })
      .def("add_sorting_constraint",
           [](Solver* s, const std::vector<IntVar*>& vars,
              const std::vector<IntVar*>& sorted) {
             s->AddConstraint(s->MakeSortingConstraint(vars, sorted));
           })
      .def("add_sub_circuit",
           [](Solver* s, const std::vector<IntVar*>& nexts) {
             s->AddConstraint(s->MakeSubCircuit(nexts));
           })
      .def("add_sum_equality",
           [](Solver* s, const std::vector<IntVar*>& vars, int64_t cst) {
             s->AddConstraint(s->MakeSumEquality(vars, cst));
           })
      .def("add_sum_equality",
           [](Solver* s, const std::vector<IntVar*>& vars, IntVar* var) {
             s->AddConstraint(s->MakeSumEquality(vars, var));
           })
      .def("add_sum_greater_or_equal",
           [](Solver* s, const std::vector<IntVar*>& vars, int64_t cst) {
             s->AddConstraint(s->MakeSumGreaterOrEqual(vars, cst));
           })
      .def("add_sum_less_or_equal",
           [](Solver* s, const std::vector<IntVar*>& vars, int64_t cst) {
             s->AddConstraint(s->MakeSumLessOrEqual(vars, cst));
           })
      .def("add_temporal_disjunction",
           [](Solver* s, IntervalVar* t1, IntervalVar* t2) {
             s->AddConstraint(s->MakeTemporalDisjunction(t1, t2));
           })
      .def("add_true_constraint",
           [](Solver* s) { s->AddConstraint(s->MakeTrueConstraint()); })
      .def("accept", &Solver::Accept, DOC(operations_research, Solver, Accept),
           py::arg("visitor"))
      .def("print_model_visitor", &Solver::MakePrintModelVisitor,
           DOC(operations_research, Solver, MakePrintModelVisitor),
           py::return_value_policy::reference_internal)
      .def("phase",
           py::overload_cast<const std::vector<IntVar*>&,
                                   Solver::IntVarStrategy,
                                   Solver::IntValueStrategy>(
               &Solver::MakePhase),
           DOC(operations_research, Solver, MakePhase),
           py::return_value_policy::reference_internal)
      .def("new_search",
           py::overload_cast<DecisionBuilder*,
                                   const std::vector<operations_research::SearchMonitor*>&>(  // NOLINT
               &Solver::NewSearch),
           DOC(operations_research, Solver, NewSearch))
      .def("new_search",
           [](Solver* s, DecisionBuilder* db) {
             std::vector<operations_research::SearchMonitor*> monitors;
             s->NewSearch(db, monitors);
           },
           DOC(operations_research, Solver, NewSearch))
      .def("next_solution", &Solver::NextSolution,
           DOC(operations_research, Solver, NextSolution))
      .def("end_search", &Solver::EndSearch,
           DOC(operations_research, Solver, EndSearch));

  py::class_<BaseObject>(m, "BaseObject", DOC(operations_research, BaseObject))
      .def("__str__", &BaseObjectPythonHelper::DebugString);

  py::class_<PropagationBaseObject, BaseObject>(
      m, "PropagationBaseObject",
      DOC(operations_research, PropagationBaseObject))
      .def_property("name", &PropagationBaseObjectPythonHelper::name,
                    &PropagationBaseObjectPythonHelper::SetName);

  // Note: no ctor.
  py::class_<IntExpr, PropagationBaseObject>(m, "IntExpr",
                                             DOC(operations_research, IntExpr))
      .def_property_readonly("min", &IntExprPythonHelper::Min,
                             DOC(operations_research, IntExpr, Min))
      .def_property_readonly("max", &IntExprPythonHelper::Max,
                             DOC(operations_research, IntExpr, Max))
      .def("set_min", &IntExprPythonHelper::SetMin,
           DOC(operations_research, IntExpr, SetMin), py::arg("m"))
      .def("set_max", &IntExprPythonHelper::SetMax,
           DOC(operations_research, IntExpr, SetMax), py::arg("m"))
      .def("set_range", &IntExprPythonHelper::SetRange,
           DOC(operations_research, IntExpr, SetRange), py::arg("mi"),
           py::arg("ma"))
      .def("set_value", &IntExprPythonHelper::SetValue,
           DOC(operations_research, IntExpr, SetValue), py::arg("v"))
      .def("bound", &IntExprPythonHelper::Bound,
           DOC(operations_research, IntExpr, Bound))
      .def("var", &IntExpr::Var, DOC(operations_research, IntExpr, Var),
           py::return_value_policy::reference_internal)
      .def(
          "__add__",
          [](IntExpr* e, int64_t arg) { return e->solver()->MakeSum(e, arg); },
          py::return_value_policy::reference_internal)
      .def(
          "__add__",
          [](IntExpr* e, IntExpr* arg) { return e->solver()->MakeSum(e, arg); },
          py::return_value_policy::reference_internal)
      .def(
          "__radd__",
          [](IntExpr* e, int64_t arg) { return e->solver()->MakeSum(e, arg); },
          py::return_value_policy::reference_internal)
      .def(
          "__radd__",
          [](IntExpr* e, IntExpr* arg) { return e->solver()->MakeSum(e, arg); },
          py::return_value_policy::reference_internal)
      .def(
          "__mul__",
          [](IntExpr* e, int64_t arg) { return e->solver()->MakeProd(e, arg); },
          py::return_value_policy::reference_internal)
      .def(
          "__mul__",
          [](IntExpr* e, IntExpr* arg) {
            return e->solver()->MakeProd(e, arg);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__rmul__",
          [](IntExpr* e, int64_t arg) { return e->solver()->MakeProd(e, arg); },
          py::return_value_policy::reference_internal)
      .def(
          "__rmul__",
          [](IntExpr* e, IntExpr* arg) {
            return e->solver()->MakeProd(e, arg);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__sub__",
          [](IntExpr* e, int64_t v) { return e->solver()->MakeSum(e, -v); },
          py::return_value_policy::reference_internal)
      .def(
          "__sub__",
          [](IntExpr* e, IntExpr* other) {
            return e->solver()->MakeDifference(e, other);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__rsub__",
          [](IntExpr* e, int64_t v) {
            return e->solver()->MakeDifference(v, e);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__rsub__",
          [](IntExpr* e, IntExpr* other) {
            return e->solver()->MakeDifference(other, e);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__floordiv__",
          [](IntExpr* e, int64_t v) { return e->solver()->MakeDiv(e, v); },
          py::return_value_policy::reference_internal)
      .def(
          "__floordiv__",
          [](IntExpr* e, IntExpr* other) {
            return e->solver()->MakeDiv(e, other);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__mod__",
          [](IntExpr* e, int64_t v) { return e->solver()->MakeModulo(e, v); },
          py::return_value_policy::reference_internal)
      .def(
          "__mod__",
          [](IntExpr* e, IntExpr* other) {
            return e->solver()->MakeModulo(e, other);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__neg__", [](IntExpr* e) { return e->solver()->MakeOpposite(e); },
          py::return_value_policy::reference_internal)
      .def(
          "__abs__", [](IntExpr* e) { return e->solver()->MakeAbs(e); },
          py::return_value_policy::reference_internal)
      .def(
          "square", [](IntExpr* e) { return e->solver()->MakeSquare(e); },
          py::return_value_policy::reference_internal)
      .def(
          "__eq__",
          [](IntExpr* left, IntExpr* right) {
            return left->solver()->MakeEquality(left, right);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__eq__",
          [](IntExpr* left, int64_t right) {
            return left->solver()->MakeEquality(left, right);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__ne__",
          [](IntExpr* left, IntExpr* right) {
            return left->solver()->MakeNonEquality(left, right);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__ne__",
          [](IntExpr* left, int64_t right) {
            return left->solver()->MakeNonEquality(left, right);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__ge__",
          [](IntExpr* left, IntExpr* right) {
            return left->solver()->MakeGreaterOrEqual(left, right);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__ge__",
          [](IntExpr* left, int64_t right) {
            return left->solver()->MakeGreaterOrEqual(left, right);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__gt__",
          [](IntExpr* left, IntExpr* right) {
            return left->solver()->MakeGreater(left, right);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__gt__",
          [](IntExpr* left, int64_t right) {
            return left->solver()->MakeGreater(left, right);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__le__",
          [](IntExpr* left, IntExpr* right) {
            return left->solver()->MakeLessOrEqual(left, right);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__le__",
          [](IntExpr* left, int64_t right) {
            return left->solver()->MakeLessOrEqual(left, right);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__lt__",
          [](IntExpr* left, IntExpr* right) {
            return left->solver()->MakeLess(left, right);
          },
          py::return_value_policy::reference_internal)
      .def(
          "__lt__",
          [](IntExpr* left, int64_t right) {
            return left->solver()->MakeLess(left, right);
          },
          py::return_value_policy::reference_internal);

  // Note: no ctor.
  py::class_<IntVar, IntExpr>(m, "IntVar", DOC(operations_research, IntVar))
      .def("value", &IntVarPythonHelper::Value,
           DOC(operations_research, IntVar, Value))
      .def("remove_value", &IntVarPythonHelper::RemoveValue,
           DOC(operations_research, IntVar, RemoveValue), py::arg("v"))
      .def("size", &IntVarPythonHelper::Size,
           DOC(operations_research, IntVar, Size));

  // Note: no ctor.
  py::class_<IntervalVar, PropagationBaseObject>(
      m, "IntervalVar", DOC(operations_research, IntervalVar))
      .def_property_readonly(
          "name", &IntervalVar::name,
          DOC(operations_research, PropagationBaseObject, name))
      .def("start_min", &IntervalVar::StartMin,
           DOC(operations_research, IntervalVar, StartMin))
      .def("start_max", &IntervalVar::StartMax,
           DOC(operations_research, IntervalVar, StartMax))
      .def("end_min", &IntervalVar::EndMin,
           DOC(operations_research, IntervalVar, EndMin))
      .def("end_max", &IntervalVar::EndMax,
           DOC(operations_research, IntervalVar, EndMax))
      .def("duration_min", &IntervalVar::DurationMin,
           DOC(operations_research, IntervalVar, DurationMin))
      .def("duration_max", &IntervalVar::DurationMax,
           DOC(operations_research, IntervalVar, DurationMax))
      .def("must_be_performed", &IntervalVar::MustBePerformed,
           DOC(operations_research, IntervalVar, MustBePerformed))
      .def("may_be_performed", &IntervalVar::MayBePerformed,
           DOC(operations_research, IntervalVar, MayBePerformed))
      .def("start_expr", &IntervalVar::StartExpr,
           DOC(operations_research, IntervalVar, StartExpr),
           py::return_value_policy::reference_internal)
      .def("duration_expr", &IntervalVar::DurationExpr,
           DOC(operations_research, IntervalVar, DurationExpr),
           py::return_value_policy::reference_internal)
      .def("end_expr", &IntervalVar::EndExpr,
           DOC(operations_research, IntervalVar, EndExpr),
           py::return_value_policy::reference_internal)
      .def("performed_expr", &IntervalVar::PerformedExpr,
           DOC(operations_research, IntervalVar, PerformedExpr),
           py::return_value_policy::reference_internal);

  // Note: no ctor.
  py::class_<Constraint>(m, "Constraint", DOC(operations_research, Constraint))
      .def("var", &Constraint::Var, DOC(operations_research, Constraint, Var));

  // Note: no ctor.
  py::class_<operations_research::SearchMonitor, BaseObject>(
      m, "SearchMonitor", DOC(operations_research, SearchMonitor));

  // Note: no ctor.
  py::class_<DecisionBuilder, BaseObject>(
      m, "DecisionBuilder", DOC(operations_research, DecisionBuilder))
      .def_property("name", &DecisionBuilder::GetName,
                    &DecisionBuilder::set_name);

  // Note: no ctor.
  py::class_<ModelVisitor, BaseObject>(m, "ModelVisitor",
                                       DOC(operations_research, ModelVisitor));

  py::class_<Assignment, PropagationBaseObject>(
      m, "Assignment", DOC(operations_research, Assignment))
      .def(py::init<Solver*>())
      .def("clear", &Assignment::Clear)
      .def("empty", &Assignment::Empty)
      .def("size", &Assignment::Size)
      .def("num_int_vars", &Assignment::NumIntVars)
      .def("num_interval_vars", &Assignment::NumIntervalVars)
      .def("num_sequence_vars", &Assignment::NumSequenceVars)
      .def("store", &Assignment::Store)
      .def("restore", &Assignment::Restore)
      .def("load", py::overload_cast<const std::string&>(&Assignment::Load),
           py::arg("filename"))
      .def("load", py::overload_cast<const AssignmentProto&>(&Assignment::Load),
           py::arg("assignment_proto"))
      .def("add_objective", &Assignment::AddObjective, py::arg("v"))
      .def("add_objectives", &Assignment::AddObjectives, py::arg("vars"))
      .def("clear_objective", &Assignment::ClearObjective)
      .def("num_objectives", &Assignment::NumObjectives)
      .def("objective", &Assignment::Objective)
      .def("objective_from_index", &Assignment::ObjectiveFromIndex,
           py::arg("index"))
      .def("has_objective", &Assignment::HasObjective)
      .def("has_objective_from_index", &Assignment::HasObjectiveFromIndex,
           py::arg("index"))
      .def("objective_min", &Assignment::ObjectiveMin)
      .def("objective_max", &Assignment::ObjectiveMax)
      .def("objective_value", &Assignment::ObjectiveValue)
      .def("objective_bound", &Assignment::ObjectiveBound)
      .def("set_objective_min", &Assignment::SetObjectiveMin, py::arg("m"))
      .def("set_objective_max", &Assignment::SetObjectiveMax, py::arg("m"))
      .def("set_objective_value", &Assignment::SetObjectiveValue,
           py::arg("value"))
      .def("set_objective_range", &Assignment::SetObjectiveRange, py::arg("l"),
           py::arg("u"))
      .def("objective_min_from_index", &Assignment::ObjectiveMinFromIndex,
           py::arg("index"))
      .def("objective_max_from_index", &Assignment::ObjectiveMaxFromIndex,
           py::arg("index"))
      .def("objective_value_from_index", &Assignment::ObjectiveValueFromIndex,
           py::arg("index"))
      .def("objective_bound_from_index", &Assignment::ObjectiveBoundFromIndex,
           py::arg("index"))
      .def("set_objective_min_from_index",
           &Assignment::SetObjectiveMinFromIndex, py::arg("index"),
           py::arg("m"))
      .def("set_objective_max_from_index",
           &Assignment::SetObjectiveMaxFromIndex, py::arg("index"),
           py::arg("m"))
      .def("set_objective_range_from_index",
           &Assignment::SetObjectiveRangeFromIndex, py::arg("index"),
           py::arg("l"), py::arg("u"))
      .def("add", py::overload_cast<IntVar*>(&Assignment::Add), py::arg("var"))
      .def("add",
           py::overload_cast<const std::vector<IntVar*>&>(&Assignment::Add),
           py::arg("var"))
      .def("min", &Assignment::Min, py::arg("var"))
      .def("max", &Assignment::Max, py::arg("var"))
      .def("value", &Assignment::Value, py::arg("var"))
      .def("bound", &Assignment::Bound, py::arg("var"))
      .def("set_min", &Assignment::SetMin, py::arg("var"), py::arg("m"))
      .def("set_max", &Assignment::SetMax, py::arg("var"), py::arg("m"))
      .def("set_range", &Assignment::SetRange, py::arg("var"), py::arg("l"),
           py::arg("u"))
      .def("set_value", &Assignment::SetValue, py::arg("var"), py::arg("value"))
      .def("add", py::overload_cast<IntervalVar*>(&Assignment::Add),
           py::arg("var"))
      .def(
          "add",
          py::overload_cast<const std::vector<IntervalVar*>&>(&Assignment::Add),
          py::arg("var"));
  // missing IntervalVar, SequenceVar, active/deactivate, contains, copy
}  // NOLINT(readability/fn_size)
