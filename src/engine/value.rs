use std::cell::RefCell;
use std::rc::Rc;
use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Div;
use std::ops::Neg;



struct ValueData {
    data: f64,
    grad: f64,
    _backward: Option<Box<dyn FnMut(f64)>>,
    _prev: Vec<Value>,
    _op: String,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(data: f64) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            _backward: None,
            _prev: vec![],
            _op: String::new(),
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn pow(&self, exponent: f64) -> Value {
        let out_data = self.data().powf(exponent);

        let out = Value(Rc::new(RefCell::new(ValueData {
            data: out_data,
            grad: 0.0,
            _backward: None,
            _prev: vec![],
            _op: "**".to_string()
        })));

        let self_clone = self.clone();
        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad: f64| {
            self_clone.0.borrow_mut().grad += exponent * self_clone.data().powf(exponent-1.0) * out_grad;
        }));

        out
    }

    pub fn inv(&self) -> Value {
        self.pow(-1.0)
    }

    pub fn tanh(&self) -> Value {
        let out_data = self.data().tanh();
        let out = Value(Rc::new(RefCell::new(ValueData {
            data: out_data,
            grad: 0.0,
            _backward: None,
            _prev: vec![self.clone()],
            _op: "tanh".to_string(),
        })));

        let self_clone = self.clone();
        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad: f64| {
            self_clone.0.borrow_mut().grad += (1.0 - out_data.powi(2)) * out_grad;
        }));

        out
    }

    pub fn relu(&self) -> Value {
        let out_data = self.data().max(0.0);
        let out = Value(Rc::new(RefCell::new(ValueData {
            data: out_data,
            grad: 0.0,
            _backward: None,
            _prev: vec![self.clone()],
            _op: "ReLU".to_string(),
        })));

        let self_clone = self.clone();
        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad: f64| {
            if self_clone.data() > 0.0 {
                self_clone.0.borrow_mut().grad += out_grad;
            }
        }));

        out
    }

    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: std::collections::HashSet<Value> = std::collections::HashSet::new();

        fn build_topo(v: &Value, visited: &mut std::collections::HashSet<Value>, topo: &mut Vec<Value>) {
            if visited.insert(v.clone()) {
                for p in v.0.borrow()._prev.iter() {
                    build_topo(p, visited, topo);
                }
                topo.push(v.clone());
            }
        }

        build_topo(self, &mut visited, &mut topo);

        self.0.borrow_mut().grad = 1.0;

        for v in topo.iter().rev() {
            let current_out_grad = v.grad();
            if let Some(mut backward_fn) = v.0.borrow_mut()._backward.take() {
                backward_fn(current_out_grad);
            }
        }
    }
}



impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}


impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        let out_data = self.data() + rhs.data();

        // Create a new node with new value
        let out = Value(Rc::new(RefCell::new(ValueData {
            data: out_data,
            grad: 0.0,
            _backward: None,
            _prev: vec![self.clone(), rhs.clone()], // Track history
            _op: "+".to_string(),
        })));

        // Define backward pass closure
        let self_clone = self.clone();
        let rhs_clone = rhs.clone();

        // Note: We use interior mutability (borrow_mut) to update gradients
        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad: f64| {
            // self.grad += 1.0 * out.grad
            self_clone.0.borrow_mut().grad += out_grad;

            // rhs.grad += 1.0 * out.grad
            rhs_clone.0.borrow_mut().grad += out_grad;
        }));

        out
    }
}

impl Add<f64> for Value {
    type Output = Value;

    fn add(self, rhs: f64) -> Value {
        self + Value::new(rhs)
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        let out_data = self.data() - rhs.data();

        // Create a new node with new value
        let out = Value(Rc::new(RefCell::new(ValueData {
            data: out_data,
            grad: 0.0,
            _backward: None,
            _prev: vec![self.clone(), rhs.clone()], // Track history
            _op: "-".to_string(),
        })));

        // Define backward pass closure
        let self_clone = self.clone();
        let rhs_clone = rhs.clone();

        // Note: We use interior mutability (borrow_mut) to update gradients
        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad: f64| {
            // self.grad += 1.0 * out.grad
            self_clone.0.borrow_mut().grad += out_grad;

            // rhs.grad -= 1.0 * out.grad
            rhs_clone.0.borrow_mut().grad -= out_grad;
        }));

        out
    }
}

impl Sub<f64> for Value {
    type Output = Value;
    fn sub(self, rhs: f64) -> Value {
        self - Value::new(rhs)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        let out_data = self.data() * rhs.data();

        // Node
        let out = Value(Rc::new(RefCell::new(ValueData {
            data: out_data,
            grad: 0.0,
            _backward: None,
            _prev: vec![self.clone(), rhs.clone()],
            _op: "*".to_string(),
        })));

        // Backward pass
        let self_clone = self.clone();
        let rhs_clone = rhs.clone();

        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad: f64| {
            self_clone.0.borrow_mut().grad += rhs_clone.data() * out_grad;
            rhs_clone.0.borrow_mut().grad += self_clone.data() * out_grad;
        }));

        out
    }
}

impl Mul<f64> for Value {
    type Output = Value;
    fn mul(self, rhs: f64) -> Value {
        self * Value::new(rhs)
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Value) -> Value {
        self * rhs.pow(-1.0)
    }
}

impl Div<f64> for Value {
    type Output = Value;
    fn div(self, rhs: f64) -> Value {
        self / Value::new(rhs)
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Value {
        self * -1.0
    }
}

impl Add<Value> for f64 {
    type Output = Value;
    fn add(self, rhs: Value) -> Value {
        rhs + self
    }
}

impl Mul<Value> for f64 {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        rhs * self
    }
}

impl Div<Value> for f64 {
    type Output = Value;
    fn div(self, rhs: Value) -> Value {
        Value::new(self) / rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_new() {
        let v = Value::new(3.14);
        assert_eq!(v.data(), 3.14);
        assert_eq!(v.grad(), 0.0);
    }

    #[test]
    fn test_value_data() {
        let v = Value::new(2.71);
        assert_eq!(v.data(), 2.71);
    }



    #[test]
    fn test_value_add() {
        let v1 = Value::new(1.0);
        let v2 = Value::new(2.0);

        let v_result = v1 + v2;
        assert_eq!(v_result.data(), 3.0);
        assert_eq!(v_result.grad(), 0.0);
    }

    #[test]
    fn test_value_add_float() {
        let v1 = Value::new(2.0);
        let v2 = 5.0;

        let v_result = v1 + v2;
        assert_eq!(v_result.data(), 7.0);
        assert_eq!(v_result.grad(), 0.0);
    }

    #[test]
    fn test_value_sub() {
        let v1 = Value::new(-3.0);
        let v2 = Value::new(-5.0);

        let v_result = v1 - v2; // -3-(-5) = -3+5=2
        assert_eq!(v_result.data(), 2.0);
        assert_eq!(v_result.grad(), 0.0);
    }

    #[test]
    fn test_value_sub_float() {
        let v1 = Value::new(-3.0);
        let v2 = -5.0;

        let v_result = v1 - v2; // -3-(-5) = -3+5=2
        assert_eq!(v_result.data(), 2.0);
        assert_eq!(v_result.grad(), 0.0);
    }

    #[test]
    fn test_value_mul() {
        let v1 = Value::new(2.0);
        let v2 = Value::new(3.0);

        let v_result = v1 * v2;
        assert_eq!(v_result.data(), 6.0);
        assert_eq!(v_result.grad(), 0.0);
    }

    #[test]
    fn test_value_mul_float() {
        let v1 = Value::new(2.0);
        let v2 = 3.0;

        let v_result = v1 * v2;
        assert_eq!(v_result.data(), 6.0);
        assert_eq!(v_result.grad(), 0.0);
    }

    #[test]
    fn test_value_pow() {
        let v = Value::new(2.0);
        let exponent: f64 = 3.0;

        let v_result = v.pow(exponent);
        assert_eq!(v_result.data(), 8.0);
        assert_eq!(v_result.grad(), 0.0);
    }

    #[test]
    fn test_value_div() {
        let v1 = Value::new(6.0);
        let v2 = Value::new(2.0);

        let v_result = v1 / v2;
        assert_eq!(v_result.data(), 3.0);
        assert_eq!(v_result.grad(), 0.0);
    }

    #[test]
    fn test_value_div_float() {
        let v1 = Value::new(6.0);
        let v2 = 2.0;

        let v_result = v1 / v2;
        assert_eq!(v_result.data(), 3.0);
        assert_eq!(v_result.grad(), 0.0);
    }

    #[test]
    fn test_value_inv() {
        let v = Value::new(2.0);

        let v_result = v.inv();
        assert_eq!(v_result.data(), 0.5);
        assert_eq!(v_result.grad(), 0.0);
    }

    #[test]
    fn test_simple_add_backward() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.clone() + b.clone();

        c.backward();

        let tol = 1e-9;
        assert!((a.grad() - 1.0).abs() < tol, "a.grad mismatch: expected 1.0, got {}", a.grad());
        assert!((b.grad() - 1.0).abs() < tol, "b.grad mismatch: expected 1.0, got {}", b.grad());
    }

    #[test]
    fn test_chained_add_backward() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c_initial = a.clone() + b.clone(); // c_initial = 5.0
        let c_final = c_initial.clone() + a.clone(); // c_final = 5.0 + 2.0 = 7.0

        c_final.backward();

        let tol = 1e-9;
        // Expected gradients:
        // d(c_final)/d(c_initial) = 1
        // d(c_final)/da = 1 (direct from c_final)
        // d(c_initial)/da = 1 (from c_initial)
        // d(c_initial)/db = 1 (from c_initial)
        // a.grad = 1 (from c_final) + 1 (from c_initial) = 2.0
        // b.grad = 1 (from c_initial) = 1.0
        assert!((a.grad() - 2.0).abs() < tol, "a.grad mismatch: expected 2.0, got {}", a.grad());
        assert!((b.grad() - 1.0).abs() < tol, "b.grad mismatch: expected 1.0, got {}", b.grad());
    }

    #[test]
    fn test_value_grad() {
        let a = Value::new(-4.0);
        let b = Value::new(2.0);

        dbg!("Initial grads:", a.grad(), b.grad());

        // Python: c = a + b
        let c_py_initial = a.clone() + b.clone();

        // Python: d = a * b + b**3
        let d_py_initial_part1 = a.clone() * b.clone();
        let d_py_initial_part2 = b.clone().pow(3.0);
        let d_py_initial = d_py_initial_part1 + d_py_initial_part2;

        // Python: c = c + c + 1  (parsed as (c + c) + 1)
        let c_py_inter1_part1 = c_py_initial.clone() + c_py_initial.clone();
        let c_py_inter1 = c_py_inter1_part1 + 1.0;

        // Python: c = 1 + c + (-a) (parsed as (1 + c) + (-a))
        let c_py_inter2_part1 = Value::new(1.0) + c_py_inter1.clone();
        let c_py_final = c_py_inter2_part1 + (-a.clone());

        // Python: d = d + d * 2 + (b + a).relu() (parsed as (d + d*2) + (b+a).relu())
        let d_py_inter1_part1 = d_py_initial.clone() + (d_py_initial.clone() * 2.0);
        let d_py_inter1_part2_arg = b.clone() + a.clone();
        let d_py_inter1_part2_res = d_py_inter1_part2_arg.relu();
        let d_py_inter1 = d_py_inter1_part1 + d_py_inter1_part2_res;

        // Python: d = d + 3 * d + (b - a).relu() (parsed as (d + 3*d) + (b-a).relu())
        let d_py_inter2_part1 = d_py_inter1.clone() + (d_py_inter1.clone() * 3.0);
        let d_py_inter2_part2_arg = b.clone() - a.clone();
        let d_py_inter2_part2_res = d_py_inter2_part2_arg.relu();
        let d_py_final = d_py_inter2_part1 + d_py_inter2_part2_res;

        // Python: e = c - d
        let e_final = c_py_final - d_py_final;

        // Python: f = e**2
        let f_final = e_final.pow(2.0);

        // Python: g = f / 2.0
        let g_inter1 = f_final.clone() / 2.0;

        // Python: g = g + 10.0 / f
        let g_inter2_part1 = Value::new(10.0) / f_final.clone();
        let g_final = g_inter1 + g_inter2_part1;

        g_final.backward();

        dbg!("Final grads:", a.grad(), b.grad());

        let tol = 1e-9;
        let a_grad_expected = 138.83333333333331;
        let b_grad_expected = 645.5624999999999;

        assert!((a.grad() - a_grad_expected).abs() < tol, "a.grad mismatch: expected {}, got {}", a_grad_expected, a.grad());
        assert!((b.grad() - b_grad_expected).abs() < tol, "b.grad mismatch: expected {}, got {}", b_grad_expected, b.grad());
    }

}