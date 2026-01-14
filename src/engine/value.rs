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
            _prev: vec![self.clone()],
            _op: "^".to_string()
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
        let mut in_progress: std::collections::HashMap<Value, bool> = std::collections::HashMap::new();

        fn build_topo(v: &Value, visited: &mut std::collections::HashSet<Value>, topo: &mut Vec<Value>, in_progress: &mut std::collections::HashMap<Value, bool>) {
            if in_progress.contains_key(v) {
                println!("ERROR");
                return;
            }
            if visited.insert(v.clone()) {
                in_progress.insert(v.clone(), true);
                for p in v.0.borrow()._prev.iter() {
                    build_topo(p, visited, topo, in_progress);
                }
                in_progress.remove(v);
                topo.push(v.clone());
            }
        }

        build_topo(self, &mut visited, &mut topo, &mut in_progress);

        self.0.borrow_mut().grad = 1.0;

        for v in topo.iter().rev() {
            println!("Processing op: {}", v.0.borrow()._op);
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
    pub fn test_simple_mul_backward() {
        let a: Value = Value::new(2.5);
        let b: Value = Value::new(3.0);
        let c = a.clone() * b.clone();

        c.backward();

        let tol = 1e-9;

        assert!((a.grad() - b.0.borrow().data).abs() < tol, "a.grad mismatch: expected {}, got {}", b.0.borrow().data, a.grad());
        assert!((b.grad() - a.0.borrow().data).abs() < tol, "b.grad mismatch: expected {}, got {}", a.0.borrow().data, b.grad());
    }

    #[test]
    fn test_chained_mul_backward() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c_initial = a.clone() * b.clone(); // c_initial = a*b = 6.0
        let c_final = c_initial.clone() * a.clone(); // c_final = c * a = a*b*a = a^2 * b = 6.0 * 2.0 = 12.0

        c_final.backward();

        let tol = 1e-9;
        // Expected gradients:
        // L = c_final = (c_initial * a) = a*b * a = a^2 *b
        // d(c_final) = dL/dc_final = 1.0
        // da = dL/da = 2ab
        // db = dL/db = a^2

        assert!((c_final.grad() - 1.0).abs() < tol, "a.grad mismatch: expected 2.0, got {}", c_final.grad());
        assert!((a.grad() - 12.0).abs() < tol, "a.grad mismatch: expected 12.0, got {}", a.grad());
        assert!((b.grad() - 4.0).abs() < tol, "a.grad mismatch: expected 4.0, got {}", b.grad());
    }

    #[test]
    pub fn test_simple_pow_backward() {
        let a: Value = Value::new(2.5);
        let b: Value = a.pow(2.0);

        b.backward();

        // b = a^2
        // da = db/da = 2a = 5.0

        let tol = 1e-9;

        assert!((a.grad() - 5.0).abs() < tol, "a.grad mismatch: expected 5.0, got {}", a.grad());
    }

    #[test]
    fn test_chained_pow_backward() {
        let a = Value::new(2.0);
        let b = 3.0;
        let c_initial = a.clone().pow(b.clone()); // c_initial = a^b = a^3
        let c_final = c_initial.clone().pow(b.clone()); // c_final = (a^b)^b = a^(b^2) = a^9

        c_final.backward();

        let tol = 1e-9;
        // Expected gradients:
        // L = c_final = a^9
        // da = dL/da = 9a^8

        assert!((a.grad() - 2304.0).abs() < tol, "a.grad mismatch: expected 2308, got {}", a.grad());
    }

    #[test]
    fn test_value_grad() {
        let a = Value::new(-4.0);
        let b = Value::new(2.0);
        let mut c = a.clone() + b.clone(); // consumes clones of a and b
        let mut d = a.clone() * b.clone() + b.clone().pow(3.0); // also consumes clones of a and b
        c = c.clone() + (c.clone() + 1.0);
        c = c.clone() + 1.0 + c.clone() + (a.clone().neg());
        d = d.clone() + (d.clone() * 2.0) + (b.clone() + a.clone()).relu();
        d = d.clone() + (d.clone() * 3.0) + (b.clone() - a.clone()).relu();
        let e = c.clone() - d.clone();
        let f = e.clone().pow(2.0);
        let mut g = f.clone() / 2.0;
        g = g + 10.0 / f;

        println!("Data of g = {}", g.data());

        g.backward();
        println!("Grad of g = {}", g.grad());

        println!("Grad of a: {}", a.grad());
        println!("Grad of b: {}", b.grad());

        assert!(true, "False");

    }

}